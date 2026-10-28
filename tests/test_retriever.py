from dataclasses import replace
from pathlib import Path
import unittest

from wow_intelligence.rag.models.knowledge_base import KnowledgeBase
from wow_intelligence.rag.pipeline import RagPipeline
from wow_intelligence.rag.retrieval.bm25 import BM25Retriever
from wow_intelligence.rag.storage.json_loader import load_knowledge_base


class RetrieverTests(unittest.TestCase):
    # Verify exact build filtering, predictable ranking, and empty results.
    def setUp(self):
        self.knowledge_base = load_knowledge_base(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json")
        self.retriever = BM25Retriever(self.knowledge_base, "test-build")

    def test_ranks_relevant_words_and_never_mixes_builds(self):
        hits = self.retriever.search("Where is the amber token?")
        self.assertEqual(hits[0].evidence_id, "amber-token")
        self.assertNotIn("amber-other-build", [item.evidence_id for item in hits])
        self.assertEqual(len(self.retriever.search("token", limit=1)), 1)

    def test_returns_no_results_for_unknown_terms_or_generic_words(self):
        for query in ("", "?!", "What is World of Warcraft Forever?", "dragon sapphire"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query), ())

    def test_filters_disabled_unapproved_and_unverified_records_before_ranking(self):
        original = self.knowledge_base.evidence[0]
        for item in (
            replace(original, source=replace(original.source, enabled=False)),
            replace(original, language="es"),
            replace(original, source=replace(original.source, enabled=False, permission_status="pending_review")),
            *[replace(original, status=status) for status in ("unknown", "unverified_for_forever", "superseded", "withheld")],
        ):
            with self.subTest(item=item):
                retriever = BM25Retriever(KnowledgeBase((item.source,), (item,)), "test-build")
                self.assertEqual(retriever.search("amber"), ())

    def test_unknown_build_has_no_fallback(self):
        self.assertEqual(BM25Retriever(self.knowledge_base, "missing-build").search("amber"), ())
        for build in ("", "unknown", " "):
            with self.subTest(build=build), self.assertRaises(ValueError):
                BM25Retriever(self.knowledge_base, build)

    def test_ties_are_stable_and_case_is_ignored(self):
        self.assertEqual(self.retriever.search("TOKEN"), self.retriever.search("token"))
        self.assertEqual(self.retriever.search("token")[0].evidence_id, "amber-token")

    def test_default_build_is_newest_recorded_not_lexicographic_order(self):
        first = replace(self.knowledge_base.evidence[0], game_build="1.60.1.9")
        second = replace(self.knowledge_base.evidence[1], game_build="1.60.1.10")
        rag = RagPipeline(KnowledgeBase(self.knowledge_base.sources, (first, second)))
        self.assertEqual(rag.game_build, "1.60.1.10")
        self.assertEqual(rag.retriever.evidence, (second,))

    def test_newer_unverified_build_never_falls_back_to_old_facts(self):
        first = replace(self.knowledge_base.evidence[0], game_build="1.60.1.9")
        second = replace(self.knowledge_base.evidence[1], game_build="1.60.1.10", status="unverified_for_forever")
        self.assertEqual(RagPipeline(KnowledgeBase(self.knowledge_base.sources, (first, second))).retriever.evidence, ())
