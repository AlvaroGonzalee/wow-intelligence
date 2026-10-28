import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from wow_intelligence.ollama import INPUT_BUDGET, OllamaError, message_size
from wow_intelligence.rag import load_corpus
from wow_intelligence.rag.context import ABSTENTION, parse_answer, prepare_context


class ContextTests(unittest.TestCase):
    # Prevent context truncation, invented references, and stale source reuse.
    def setUp(self):
        self.evidence = load_corpus(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json").evidence[:2]
        self.history = [
            {"role": "system", "content": "No sources connected."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello."},
        ]

    def test_preserves_history_and_keeps_evidence_separate_from_instructions(self):
        before = copy.deepcopy(self.history)
        prepared = prepare_context(self.history, "Where is the amber token?", self.evidence)
        self.assertEqual(self.history, before)
        self.assertEqual(prepared.messages[1:3], before[1:])
        self.assertNotIn("No sources connected", prepared.messages[0]["content"])
        self.assertEqual(prepared.messages[-2]["role"], "user")
        self.assertIn("amber-token", prepared.messages[-2]["content"])
        self.assertNotIn("https://", prepared.messages[-2]["content"])
        self.assertEqual(prepared.messages[-1]["content"], "Where is the amber token?")
        self.assertLessEqual(message_size(prepared.messages), INPUT_BUDGET)

    def test_skips_oversized_excerpts_without_truncating_them(self):
        huge = replace(self.evidence[0], content="🐉" * 1000)
        prepared = prepare_context(self.history, "token", (huge, self.evidence[1]))
        self.assertEqual(prepared.evidence, (self.evidence[1],))
        self.assertNotIn("🐉", str(prepared.messages))
        with self.assertRaisesRegex(OllamaError, "does not fit"):
            prepare_context(self.history, "token", (huge,))

    def test_does_not_silently_drop_long_history(self):
        self.history.append({"role": "user", "content": "x" * 3000})
        with self.assertRaises(OllamaError):
            prepare_context(self.history, "token", self.evidence)

    def test_limits_context_to_three_complete_excerpts(self):
        candidates = tuple(replace(self.evidence[0], evidence_id=f"item-{n}") for n in range(6))
        prepared = prepare_context([], "token", candidates)
        self.assertEqual(len(prepared.evidence), 3)

    def test_renders_only_cited_metadata_in_first_citation_order(self):
        answer = parse_answer(json.dumps({"statements": [
            {"text": "The violet token is in a green chest.", "evidence_ids": ["violet-token"]},
            {"text": "The amber token is in a blue chest.", "evidence_ids": ["amber-token"]},
        ]}), self.evidence)
        self.assertIn("green chest. [1]", answer.text)
        self.assertIn("blue chest. [2]", answer.text)
        self.assertTrue(answer.sources_text().startswith("Sources:\n[1] violet-token"))
        self.assertIn("Build: test-build", answer.sources_text())
        self.assertIn("Verified: 2026-01-02", answer.sources_text())

    def test_preliminary_notice_depends_on_cited_evidence(self):
        preliminary = replace(self.evidence[0], status="source_reported", verified_at=None)
        raw = json.dumps({"statements": [
            {"text": "The catalog lists the token in a blue chest.", "evidence_ids": [preliminary.evidence_id]},
        ]})
        answer = parse_answer(raw, (preliminary, self.evidence[1]))
        self.assertIn("This answer is preliminary", answer.text)
        self.assertIn("may change", answer.text)
        self.assertIn("not independently verified", answer.sources_text())
        from wow_intelligence.language import EnglishLanguagePolicy
        self.assertTrue(EnglishLanguagePolicy().is_english(answer.text))
        verified_answer = parse_answer(json.dumps({"statements": [
            {"text": "The violet token is in a green chest.", "evidence_ids": [self.evidence[1].evidence_id]},
        ]}), (preliminary, self.evidence[1]))
        self.assertNotIn("This answer is preliminary", verified_answer.text)

    def test_preliminary_evidence_does_not_force_an_unsupported_answer(self):
        preliminary = replace(self.evidence[0], status="source_reported", verified_at=None)
        answer = parse_answer('{"statements":[]}', (preliminary,))
        self.assertEqual(answer.text, ABSTENTION)
        self.assertFalse(answer.evidence)

    def test_empty_statements_abstain_without_citations(self):
        answer = parse_answer('{"statements":[]}', self.evidence)
        self.assertEqual(answer.text, ABSTENTION)
        self.assertEqual(answer.evidence, ())

    def test_conflicting_catalog_fields_block_generation_context(self):
        item = replace(self.evidence[0], url="https://example.invalid/items/123-test",
                       content="Fictional items catalog reports name: Test Item; required level: 60; class: Armor;")
        conflict = replace(item, evidence_id="conflict", content=item.content.replace("level: 60", "level: 1"))
        prepared = prepare_context([], "What level does Test Item require?", (item, conflict))
        self.assertFalse(prepared.evidence)
        self.assertIn("sources disagree", prepared.notice)
        self.assertNotIn("required level", str(prepared.messages))

    def test_unknown_fields_and_distinct_item_ids_do_not_create_false_conflicts(self):
        item = replace(self.evidence[0], url="https://example.invalid/items/123-test",
                       content="Fictional items catalog reports name: Test Item; required level: 60; class: Armor;")
        other = replace(item, evidence_id="other", url="https://example.invalid/items/456-other",
                        content=item.content.replace("level: 60", "level: 1"))
        unknown = replace(item, evidence_id="unknown", content=item.content.replace("level: 60", "level: unknown"))
        prepared = prepare_context([], "What level does Test Item require?", (item, other, unknown))
        self.assertIsNone(prepared.notice)
        self.assertEqual(len(prepared.evidence), 3)

    def test_rejects_unstructured_output_and_duplicate_json_keys(self):
        for raw in ("Here is an answer.", '[]', '{"statements":null}',
                    '{"statements":[],"extra":true}', '{"statements":[],"statements":[]}'):
            with self.subTest(raw=raw), self.assertRaises(OllamaError):
                parse_answer(raw, self.evidence)

    def test_rejects_missing_fabricated_and_unselected_citations(self):
        for identifiers in ([], ["invented"], ["amber-other-build"], [123], "amber-token", ["amber-token", "amber-token"]):
            with self.subTest(identifiers=identifiers), self.assertRaises(OllamaError):
                parse_answer(json.dumps({"statements": [
                    {"text": "An answer.", "evidence_ids": identifiers},
                ]}), self.evidence)

    def test_rejects_model_urls_citation_markers_and_control_characters(self):
        for text in ("", "See https://invented.invalid", "Answer [7]", "Hello\nSource: fake", "\u001b[31m", "\u009b31m"):
            with self.subTest(text=text), self.assertRaises(OllamaError):
                parse_answer(json.dumps({"statements": [
                    {"text": text, "evidence_ids": ["amber-token"]},
                ]}), self.evidence)
