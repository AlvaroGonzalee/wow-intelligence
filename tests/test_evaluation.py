from pathlib import Path
from unittest.mock import patch
import copy
import io
import json
import tempfile
import unittest

from wow_intelligence.cli.evaluate import main
from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.rag.evaluation.benchmark import evaluate
from wow_intelligence.rag.evaluation.reference_questions import load_questions
from wow_intelligence.rag.pipeline import RagPipeline
from wow_intelligence.rag.storage.json_loader import load_knowledge_base


FIXTURE = Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json"


class EvaluationTests(unittest.TestCase):
    # Keep benchmark failures visible and separate retrieval scores from truth.
    def setUp(self):
        self.pipeline = RagPipeline(load_knowledge_base(FIXTURE), "test-build")
        self.suite = {
            "schema_version": 1, "name": "Fictional evaluation", "game_build": "test-build",
            "limitations": "Fictional test data, not real game knowledge.",
            "cases": [
                {"id": "known", "category": "known", "question": "Where is the amber token stored?",
                 "expected": "answer", "references": {"amber-token": ["blue chest"]}, "answer_terms": [["blue"]]},
                {"id": "missing", "category": "unknown", "question": "Unobtainium?",
                 "expected": "abstain", "references": {}, "answer_terms": []},
            ],
        }

    def test_retrieval_only_scores_context_without_inventing_answer_metrics(self):
        report = evaluate(self.pipeline, self.suite)
        self.assertEqual(report["metrics"]["candidate_recall_at_12"], 1)
        self.assertEqual(report["metrics"]["context_recall"], 1)
        self.assertIsNone(report["metrics"]["behavior_match_rate"])
        self.assertIsNone(report["metrics"]["answer_accuracy"])
        self.assertNotIn("amber-other-build", report["cases"][0]["ranked_ids"])
        self.assertIsNone(report["cases"][0]["human_review"]["groundedness"])

    def test_disabled_references_count_as_misses_instead_of_being_skipped(self):
        with patch.object(self.pipeline.retriever, "search", return_value=()):
            report = evaluate(self.pipeline, self.suite)
        self.assertEqual(report["metrics"]["answer_cases"], 1)
        self.assertEqual(report["metrics"]["context_recall"], 0)

    def test_stale_or_missing_references_fail_before_generation(self):
        for change in ("build", "id", "content"):
            suite = copy.deepcopy(self.suite)
            if change == "build":
                suite["game_build"] = "other-test-build"
            elif change == "id":
                suite["cases"][0]["references"] = {"absent": ["blue chest"]}
            else:
                suite["cases"][0]["references"]["amber-token"] = ["red chest"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate(self.pipeline, suite, generator=lambda *args, **kwargs: self.fail("Must not generate"))

    def test_generation_records_citations_and_skips_model_for_empty_retrieval(self):
        calls = []
        def provider(messages, *, response_schema):
            calls.append(messages)
            self.assertNotIn("blue chest", messages[-1]["content"])
            return json.dumps({"statements": [{"text": "The synthetic amber token is stored in a blue chest.", "evidence_ids": ["amber-token"]}]})
        report = evaluate(self.pipeline, self.suite, generator=provider)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["metrics"]["behavior_match_rate"], 1)
        self.assertTrue(report["cases"][0]["generation"]["reference_citations_present"])
        self.assertTrue(report["cases"][1]["generation"]["abstained"])
        self.assertIsNone(report["metrics"]["answer_accuracy"])

    def test_generation_errors_remain_in_denominator(self):
        def provider(*args, **kwargs):
            raise OllamaError("Simulated invalid citation")
        report = evaluate(self.pipeline, self.suite, generator=provider)
        self.assertEqual(report["metrics"]["errors"], 1)
        self.assertEqual(report["metrics"]["behavior_match_rate"], 0.5)
        self.assertEqual(report["cases"][0]["error"], "Simulated invalid citation")

    def test_comparison_never_supplies_evidence_to_the_baseline(self):
        baseline_calls = []
        def provider(messages, *, response_schema):
            if response_schema is None:
                baseline_calls.append(messages)
                self.assertNotIn("blue chest", str(messages))
                return "I don't know based on the verified WoW Forever sources currently available to me."
            return json.dumps({"statements": [{"text": "The synthetic amber token is stored in a blue chest.", "evidence_ids": ["amber-token"]}]})
        report = evaluate(self.pipeline, self.suite, generator=provider, compare_without_rag=True)
        self.assertEqual(len(baseline_calls), 2)
        self.assertEqual(report["metrics"]["without_rag"]["abstention_phrase_rate"], 1)
        self.assertEqual(report["metrics"]["without_rag"]["reference_term_match_rate"], 0)
        self.assertEqual(report["metrics"]["reference_term_match_rate"], 1)

    def test_loader_rejects_ambiguous_or_incomplete_references(self):
        invalid = []
        for field, value in (("expected", "maybe"), ("references", {}), ("answer_terms", [])):
            suite = copy.deepcopy(self.suite)
            suite["cases"][0][field] = value
            invalid.append(suite)
        duplicate = copy.deepcopy(self.suite)
        duplicate["cases"].append(duplicate["cases"][0])
        invalid.append(duplicate)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "questions.json"
            for suite in invalid:
                path.write_text(json.dumps(suite))
                with self.assertRaises(ValueError):
                    load_questions(path)

    def test_cli_offline_does_not_load_ollama_and_preserves_existing_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            questions = Path(folder) / "questions.json"
            output = Path(folder) / "report.json"
            questions.write_text(json.dumps(self.suite))
            with patch("sys.argv", ["evaluation", "--corpus", str(FIXTURE), "--questions", str(questions), "--output", str(output)]):
                with patch('wow_intelligence.cli.evaluate.RagPipeline', return_value=self.pipeline):
                    with patch("wow_intelligence.cli.evaluate.OllamaConfig.from_env") as config:
                        with patch("sys.stdout", new_callable=io.StringIO):
                            main()
                        config.assert_not_called()
                        before = output.read_bytes()
                        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                            main()
                        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
