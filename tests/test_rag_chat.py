import io
import json
import unittest
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from wow_intelligence.chat import run_chat
from wow_intelligence.rag import Corpus, RagPipeline, load_corpus
from wow_intelligence.rag.context import ABSTENTION


class RagChatTests(unittest.TestCase):
    # Cover real loading, retrieval, transport payloads, validation, and display.
    def setUp(self):
        corpus = load_corpus(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json")
        self.rag = RagPipeline(corpus, "test-build")
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(redirect_stderr(io.StringIO()))
        self.input = self.enterContext(patch("builtins.input"))
        self.http = self.enterContext(patch("wow_intelligence.ollama.urlopen"))
        self.language = self.enterContext(patch("wow_intelligence.chat.EnglishLanguagePolicy")).return_value
        self.language.is_english.return_value = True
        self.language.output_rejection = "Non-English response blocked."

    def response(self, identifiers=None, text="The synthetic amber token is stored in a blue chest."):
        generated = json.dumps({"statements": [
            {"text": text, "evidence_ids": identifiers if identifiers is not None else ["amber-token"]},
        ]})
        return io.BytesIO(json.dumps({"done": True, "message": {"role": "assistant", "content": generated}}).encode())

    def test_full_flow_uses_retrieved_context_and_displays_corpus_citations(self):
        self.input.side_effect = ["Where is the amber token?", "/sources", "/exit"]
        self.http.return_value = self.response()
        run_chat(rag=self.rag)
        payload = json.loads(self.http.call_args.args[0].data)
        self.assertEqual(payload["format"]["required"], ["statements"])
        self.assertEqual(payload["options"]["temperature"], 0)
        allowed = payload["format"]["properties"]["statements"]["items"]["properties"]["evidence_ids"]["items"]["enum"]
        self.assertNotIn("amber-other-build", allowed)
        self.assertIn("amber-token", payload["messages"][-2]["content"])
        self.assertNotIn("amber-other-build", str(payload))
        self.assertIn("blue chest. [1]", self.output.getvalue())
        self.assertEqual(self.output.getvalue().count("https://example.invalid/synthetic/amber"), 2)
        self.http.assert_called_once()

    def test_no_matches_abstain_without_calling_ollama(self):
        self.input.side_effect = ["Where is the sapphire dragon?", "/sources", "/exit"]
        run_chat(rag=self.rag)
        self.http.assert_not_called()
        self.assertIn(ABSTENTION, self.output.getvalue())
        self.assertIn("No sources were cited", self.output.getvalue())

    def test_clear_discards_history_citations_and_followup_subject(self):
        self.input.side_effect = ["Where is the amber token?", "/clear", "/sources", "Where is it?", "/exit"]
        self.http.return_value = self.response()
        run_chat(rag=self.rag)
        self.http.assert_called_once()
        self.assertIn("Conversation cleared.", self.output.getvalue())
        self.assertIn("No sources were cited", self.output.getvalue())
        self.assertIn("Which item, quest, class, or location", self.output.getvalue())

    def test_conflicting_catalog_records_never_call_the_model(self):
        item = replace(self.rag.corpus.evidence[0], url="https://example.invalid/items/123-test",
                       content="Fictional items catalog reports name: Test Item; required level: 60; class: Armor;")
        conflict = replace(item, evidence_id="conflict", content=item.content.replace("level: 60", "level: 1"))
        rag = RagPipeline(Corpus(self.rag.corpus.sources, (item, conflict)), "test-build")
        self.input.side_effect = ["What required level does Test Item have?", "/exit"]
        run_chat(rag=rag)
        self.http.assert_not_called()
        self.assertIn("sources disagree", self.output.getvalue())

    def test_greeting_and_thanks_work_without_retrieval_or_model_calls(self):
        self.input.side_effect = ["hello", "thanks", "/coverage", "/exit"]
        with patch.object(self.rag, "prepare") as prepare:
            run_chat(rag=self.rag)
        prepare.assert_not_called()
        self.http.assert_not_called()
        self.assertIn("Hello!", self.output.getvalue())
        self.assertIn("You're welcome!", self.output.getvalue())
        self.assertIn("Eligible evidence: 2.", self.output.getvalue())
        self.assertNotIn(ABSTENTION, self.output.getvalue())

    def test_reference_without_topic_requests_clarification(self):
        self.input.side_effect = ["Where can I find it?", "/exit"]
        run_chat(rag=self.rag)
        self.http.assert_not_called()
        self.assertIn("Which item, quest, class, or location", self.output.getvalue())

    def test_followup_retrieves_previous_topic_again(self):
        self.input.side_effect = ["Where is amber?", "thanks", "Where can I find it?", "/exit"]
        self.http.side_effect = [self.response(), self.response()]
        run_chat(rag=self.rag)
        second = json.loads(self.http.call_args_list[1].args[0].data)["messages"]
        self.assertIn("amber-token", second[-2]["content"])
        self.assertNotIn("violet-token", second[-2]["content"])

    def test_unanswered_new_topic_does_not_reuse_an_old_subject(self):
        self.input.side_effect = ["Where is amber?", "Where is sapphire?", "Where can I find it?", "/exit"]
        self.http.return_value = self.response()
        run_chat(rag=self.rag)
        self.http.assert_called_once()
        self.assertIn("Which item, quest, class, or location", self.output.getvalue())

    def test_wrong_build_abstains_without_calling_ollama(self):
        self.rag = RagPipeline(load_corpus(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json"), "missing-build")
        self.input.side_effect = ["Where is the amber token?", "/exit"]
        run_chat(rag=self.rag)
        self.http.assert_not_called()
        self.assertIn(ABSTENTION, self.output.getvalue())

    def test_invalid_citations_are_not_saved_or_displayed(self):
        self.input.side_effect = ["Where is the amber token?", "Where is the amber token?", "/exit"]
        self.http.side_effect = [self.response(["invented"], "Invented answer."), self.response()]
        run_chat(rag=self.rag)
        second = json.loads(self.http.call_args_list[1].args[0].data)
        self.assertEqual(len(second["messages"]), 3)
        self.assertNotIn("Invented answer", self.output.getvalue())

    def test_non_english_generated_answer_has_no_citations_or_history(self):
        self.input.side_effect = ["Where is the amber token?", "/sources", "/exit"]
        self.http.return_value = self.response(text="Respuesta rechazada.")
        self.language.is_english.side_effect = [True, False]
        run_chat(rag=self.rag)
        self.assertNotIn("Respuesta rechazada", self.output.getvalue())
        self.assertNotIn("https://example.invalid", self.output.getvalue())
        self.assertIn("No sources were cited", self.output.getvalue())

    def test_retrieved_context_is_not_stored_and_old_evidence_cannot_be_cited(self):
        self.input.side_effect = ["Where is amber?", "Where is violet?", "/sources", "/exit"]
        self.http.side_effect = [self.response(), self.response(["amber-token"], "Stale citation.")]
        run_chat(rag=self.rag)
        second = json.loads(self.http.call_args_list[1].args[0].data)["messages"]
        self.assertEqual(len(second), 5)
        self.assertEqual(sum("Retrieved evidence" in item["content"] for item in second), 1)
        self.assertNotIn("amber-token", second[-2]["content"])
        self.assertNotIn("Stale citation", self.output.getvalue())
        self.assertIn("[1] amber-token", self.output.getvalue())
