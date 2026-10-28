from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
import re

from wow_intelligence.chat.policies.language import EnglishLanguagePolicy
from wow_intelligence.chat.prompts import AssistantInstructions
from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.rag.generation.catalog_answers import catalog_names
from wow_intelligence.rag.pipeline import RagPipeline


def validate_references(pipeline: RagPipeline, suite: dict) -> None:
    if suite["game_build"] != pipeline.game_build:
        raise ValueError("Question references target a different build. Review them before evaluating this snapshot.")
    records = {item.evidence_id: item for item in pipeline.knowledge_base.evidence}
    for case in suite["cases"]:
        for identifier, fragments in case["references"].items():
            item = records.get(identifier)
            if item is None or item.game_build != pipeline.game_build:
                raise ValueError(f"Missing or mismatched reference: {identifier}.")
            if any(fragment not in item.content for fragment in fragments):
                raise ValueError(f"Reference content changed: {identifier}. Review expected answers.")


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(pipeline: RagPipeline, suite: dict, *, generator=None, compare_without_rag: bool = False, progress=None) -> dict:
    if compare_without_rag and generator is None:
        raise ValueError("Comparison requires a generator.")
    validate_references(pipeline, suite)
    language = EnglishLanguagePolicy() if generator is not None else None
    results = []
    for case in suite["cases"]:
        started = perf_counter()
        expected = set(case["references"])
        row = {
            "id": case["id"], "category": case["category"], "question": case["question"],
            "expected": case["expected"], "reference_ids": sorted(expected),
            "ranked_ids": [], "context_ids": [], "candidate_recall_at_12": None,
            "context_recall": None, "reciprocal_rank": None,
            "generation": None, "error": None,
            "without_rag": None,
            "human_review": {"correctness": None, "groundedness": None, "notes": ""},
        }
        try:
            prepared = pipeline.prepare([], case["question"])
            row["ranked_ids"] = list(prepared.ranked_ids)
            row["context_ids"] = [item.evidence_id for item in prepared.evidence]
            if expected:
                row["candidate_recall_at_12"] = len(expected.intersection(row["ranked_ids"])) / len(expected)
                row["context_recall"] = len(expected.intersection(row["context_ids"])) / len(expected)
                row["reciprocal_rank"] = next((1 / rank for rank, key in enumerate(row["ranked_ids"], 1) if key in expected), 0)
            row["retrieval_seconds"] = perf_counter() - started
            if generator is not None:
                answer = pipeline.answer(prepared, case["question"], generator=generator)
                text = answer.text
                cited = [item.evidence_id for item in answer.evidence]
                english = language.is_english(answer.validation_text or text, known_names=catalog_names(answer.evidence))
                accepted = english and bool(cited)
                matches = [any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE)
                               for term in group) for group in case["answer_terms"]]
                row["generation"] = {
                    "text": text, "cited_ids": cited, "english": english,
                    "abstained": english and not cited, "reference_term_matches": matches,
                    "behavior_matches_expectation": english and (accepted if expected else not cited),
                    "reference_citations_present": expected.issubset(cited) if expected else None,
                    "preliminary_notice_present": "This answer is preliminary" in text if accepted else None,
                    "answer_method": answer.method,
                }
        except OllamaError as exc:
            row["error"] = str(exc)
        row["elapsed_seconds"] = perf_counter() - started
        if compare_without_rag:
            baseline_started = perf_counter()
            baseline = {"text": None, "english": None, "abstention_phrase_detected": None,
                        "reference_term_matches": [], "error": None}
            try:
                text = generator([AssistantInstructions().system_message(), {"role": "user", "content": case["question"]}], response_schema=None)
                baseline.update(
                    text=text, english=language.is_english(text),
                    abstention_phrase_detected=bool(re.search(r"i (?:don't|do not) know|(?:not|don't have) enough information|can(?:not|'t) verify", text, re.IGNORECASE)),
                    reference_term_matches=[any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE)
                                               for term in group) for group in case["answer_terms"]],
                )
            except OllamaError as exc:
                baseline["error"] = str(exc)
            baseline["elapsed_seconds"] = perf_counter() - baseline_started
            row["without_rag"] = baseline
        results.append(row)
        if progress:
            progress(len(results), len(suite["cases"]), case["id"])
    positives = [row for row in results if row["reference_ids"]]
    generation = [row["generation"] for row in results if row["generation"] is not None]
    errors = sum(row["error"] is not None for row in results)
    return {
        "schema_version": 1, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite["name"], "game_build": pipeline.game_build,
        "include_catalog": pipeline.include_catalog, "mode": "generation" if generator else "retrieval",
        "retrieval_mode": pipeline.retrieval_mode,
        "limitations": suite["limitations"],
        "scope": "Independent RAG turns; CLI input policies and multi-turn dialogue are tested separately.",
        "metrics": {
            "cases": len(results), "categories": dict(Counter(row["category"] for row in results)),
            "answer_cases": len(positives), "abstention_cases": len(results) - len(positives), "errors": errors,
            "candidate_recall_at_12": _mean([row["candidate_recall_at_12"] or 0 for row in positives]),
            "context_recall": _mean([row["context_recall"] or 0 for row in positives]),
            "mean_reciprocal_rank": _mean([row["reciprocal_rank"] or 0 for row in positives]),
            "generation_completed": len(generation) if generator else None,
            "behavior_match_rate": sum(row["behavior_matches_expectation"] for row in generation) / len(results) if generator else None,
            "reference_term_match_rate": sum(bool(row["generation"]) and all(row["generation"]["reference_term_matches"]) for row in positives) / len(positives) if generator and positives else None,
            "answer_accuracy": None, "groundedness": None,
            "mean_elapsed_seconds": _mean([row["elapsed_seconds"] for row in results]),
            "without_rag": {
                "errors": sum(row["without_rag"]["error"] is not None for row in results),
                "abstention_phrase_rate": sum(bool(row["without_rag"]["abstention_phrase_detected"]) for row in results) / len(results),
                "reference_term_match_rate": sum(row["without_rag"]["error"] is None and all(row["without_rag"]["reference_term_matches"]) for row in positives) / len(positives) if positives else None,
                "mean_elapsed_seconds": _mean([row["without_rag"]["elapsed_seconds"] for row in results]),
            } if compare_without_rag else None,
        },
        "cases": results,
    }
