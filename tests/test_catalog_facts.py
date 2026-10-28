from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from wow_intelligence.language import EnglishLanguagePolicy
from wow_intelligence.rag import Corpus, RagPipeline, load_corpus
from wow_intelligence.rag.catalog_facts import catalog_answer


class CatalogFactsTests(unittest.TestCase):
    # Answer catalog fields from cited records without asking the LLM to guess.
    def setUp(self):
        corpus = load_corpus(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json")
        self.first = replace(corpus.evidence[0], url="https://example.invalid/items/123-first",
                             status="source_reported", verified_at=None,
                             content="Fictional items catalog reports name: Amber Token; quality: epic; item level: 71; required level: 60; class: Armor; subclass: Leather (419). Missing fields are unknown.")
        self.second = replace(self.first, evidence_id="second", url="https://example.invalid/items/456-second",
                              content=self.first.content.replace("Amber Token", "Matriarch Quiver").replace("level: 60", "level: 35"))
        self.pipeline = RagPipeline(Corpus(corpus.sources, (self.first, self.second)), "test-build", include_catalog=True)

    def test_quality_and_item_level_are_extracted_without_generated_adjectives(self):
        for question, expected in (("How rare is Amber Token?", "quality: epic"),
                                   ("How powerful is Amber Token according to its item level?", "item level: 71")):
            with self.subTest(question=question):
                provider = Mock(side_effect=AssertionError("No model call expected"))
                answer = self.pipeline.answer(self.pipeline.prepare([], question), question, generator=provider)
                self.assertEqual(answer.method, "catalog")
                self.assertIn(expected, answer.text)
                self.assertIn("This answer is preliminary", answer.text)
                self.assertEqual(answer.evidence, (self.first,))
                self.assertNotIn("powerful", answer.text)

    def test_comparison_keeps_both_records_and_partial_answer_marks_missing_drops(self):
        question = "Compare the required character levels of Amber Token and Matriarch Quiver."
        answer = catalog_answer(question, (self.first, self.second))
        self.assertEqual(len(answer.evidence), 2)
        self.assertIn("60", answer.text)
        self.assertIn("35", answer.text)
        partial = catalog_answer("What is the required level of Amber Token and which boss drops it?", (self.first,))
        self.assertIn("60", partial.text)
        self.assertIn("does not establish drop sources", partial.text)

    def test_unknown_is_not_replaced_with_a_number_and_entity_must_match(self):
        unknown = replace(self.first, content=self.first.content.replace("level: 60", "level: unknown"))
        answer = catalog_answer("What is the required level of Amber Token?", (unknown,))
        self.assertIn("required level: unknown", answer.text)
        self.assertIsNone(catalog_answer("What required level does an invented token have?", (self.first,)))
        self.assertIsNone(catalog_answer("Which boss drops Amber Token?", (self.first,)))

    def test_longer_item_name_does_not_get_shorter_names_fields(self):
        longer = replace(self.first, evidence_id="longer", content=self.first.content.replace("Amber Token", "Goo-Covered Amber Token").replace("level: 60", "level: 55"))
        answer = catalog_answer("What required level does Goo-Covered Amber Token have?", (self.first, longer))
        self.assertEqual(answer.evidence, (longer,))
        self.assertIn("55", answer.text)

    def test_other_editions_and_unavailable_builds_do_not_reach_generation(self):
        for question in ("What required level does Amber Token have in Retail?",
                         "What required level does Amber Token have in build 1.60.2.99999?"):
            with self.subTest(question=question):
                prepared = self.pipeline.prepare([], question)
                self.assertFalse(prepared.evidence)
                self.assertIn("requested edition or build", prepared.notice)
                answer = self.pipeline.answer(prepared, question, generator=Mock(side_effect=AssertionError()))
                self.assertEqual(answer.method, "abstention")

    def test_game_names_do_not_hide_spanish_input_or_output(self):
        policy = EnglishLanguagePolicy()
        names = self.pipeline.language_names
        self.assertTrue(policy.is_english("What character level is required for Matriarch Quiver according to the catalog?", known_names=names))
        self.assertTrue(policy.is_english("What exact character level must I reach to equip Apprentice's Mining Pack?", known_names=names + ("Apprentice's Mining Pack",)))
        self.assertFalse(policy.is_english("¿Qué nivel necesito para Matriarch Quiver?", known_names=names))
        self.assertFalse(policy.is_english("Please tell me about Matriarch Quiver. Necesito saber dónde conseguirlo.", known_names=names))
        answer = catalog_answer("What required level does Matriarch Quiver have?", (self.second,))
        self.assertTrue(policy.is_english(answer.validation_text, known_names=names))


if __name__ == "__main__":
    unittest.main()
