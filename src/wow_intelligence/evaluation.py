import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

from wow_intelligence.config import OllamaConfig
from wow_intelligence.language import EnglishLanguagePolicy
from wow_intelligence.instructions import AssistantInstructions
from wow_intelligence.ollama import OllamaError, generate
from wow_intelligence.rag import RagPipeline, load_corpus
from wow_intelligence.rag.context import RAG_INSTRUCTIONS
from wow_intelligence.rag.corpus import _unique_object
from wow_intelligence.rag.embedder import OllamaEmbedder
from wow_intelligence.rag.catalog_facts import catalog_names


def load_questions(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Evaluation questions exceed 1 MiB.")
    suite = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(suite, dict) or set(suite) != {"schema_version", "name", "game_build", "limitations", "cases"}:
        raise ValueError("Invalid evaluation suite fields.")
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("Unsupported evaluation schema version.")
    if any(not isinstance(suite[key], str) or not suite[key].strip() for key in ("name", "game_build", "limitations")):
        raise ValueError("Evaluation metadata must contain nonempty text.")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("Evaluation requires at least one case.")
    seen = set()
    for case in suite["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "category", "question", "expected", "references", "answer_terms"}:
            raise ValueError("Invalid evaluation case fields.")
        if any(not isinstance(case[key], str) or not case[key].strip() for key in ("id", "category", "question")):
            raise ValueError("Cases require nonempty identifiers, categories and questions.")
        if case["id"] in seen:
            raise ValueError("Duplicate evaluation case ID.")
        seen.add(case["id"])
        if case["expected"] not in ("answer", "abstain") or not isinstance(case["references"], dict):
            raise ValueError("Cases require answer/abstain and a reference mapping.")
        if (case["expected"] == "answer") != bool(case["references"]):
            raise ValueError("Answer cases need references; abstention cases must have none.")
        if any(not isinstance(key, str) or not key or not isinstance(fragments, list) or not fragments
               or any(not isinstance(fragment, str) or not fragment for fragment in fragments)
               for key, fragments in case["references"].items()):
            raise ValueError("Each reference requires an evidence ID and nonempty source fragments.")
        groups = case["answer_terms"]
        if not isinstance(groups, list) or (case["expected"] == "answer" and not groups) or any(not isinstance(group, list) or not group
                                               or any(not isinstance(term, str) or not term for term in group)
                                               for group in groups):
            raise ValueError("Answer terms must be lists of nonempty alternative strings.")
    return suite


def validate_references(pipeline: RagPipeline, suite: dict) -> None:
    if suite["game_build"] != pipeline.game_build:
        raise ValueError("Question references target a different build. Review them before evaluating this snapshot.")
    records = {item.evidence_id: item for item in pipeline.corpus.evidence}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval; optionally call Ollama for answer review.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; existing reports are never overwritten.")
    parser.add_argument("--include-catalog", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Call configured Ollama; otherwise evaluate retrieval offline.")
    parser.add_argument("--compare-without-rag", action="store_true", help="Also run the same questions using the application's no-source instructions.")
    parser.add_argument("--index", type=Path, help="Evaluate hybrid retrieval using this vector index.")
    parser.add_argument("--embedding-model", help="Installed embedding model used to build the index.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run selected case IDs only; repeat as needed.")
    args = parser.parse_args()
    if args.compare_without_rag and not args.generate:
        parser.error("--compare-without-rag requires --generate.")
    if bool(args.index) != bool(args.embedding_model):
        parser.error("--index and --embedding-model must be supplied together.")
    try:
        if args.output.exists():
            raise ValueError("Output already exists. Choose a new report path.")
        suite = load_questions(args.questions)
        if args.case_ids:
            unknown = set(args.case_ids) - {case["id"] for case in suite["cases"]}
            if unknown:
                raise ValueError("Unknown case IDs: " + ", ".join(sorted(unknown)))
            suite["cases"] = [case for case in suite["cases"] if case["id"] in args.case_ids]
        config = OllamaConfig.from_env() if args.generate or args.index else None
        embedder = OllamaEmbedder(args.embedding_model, base_url=config.base_url, timeout=config.timeout) if args.index else None
        pipeline = RagPipeline(load_corpus(args.corpus), include_catalog=args.include_catalog,
                               index_path=args.index, embedder=embedder)
        def configured_generate(messages, *, response_schema):
            return generate(messages, response_schema=response_schema, model=config.model,
                            base_url=config.base_url, timeout=config.timeout)
        report = evaluate(pipeline, suite, generator=configured_generate if args.generate else None,
                          compare_without_rag=args.compare_without_rag,
                          progress=lambda completed, total, key: print(f"Evaluated {completed}/{total}: {key}", flush=True))
        report["model"] = config.model if config else None
        report["python_version"] = sys.version.split()[0]
        report["rag_instructions_sha256"] = hashlib.sha256(RAG_INSTRUCTIONS.encode()).hexdigest()
        report["corpus_sha256"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        report["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, OSError, OllamaError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["metrics"], indent=2))
    print(f"Report: {args.output}")
    print("Retrieval and term matches do not prove answer correctness. Review generated claims against cited sources.")
    if report["metrics"]["errors"] or (report["metrics"]["without_rag"] and report["metrics"]["without_rag"]["errors"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
