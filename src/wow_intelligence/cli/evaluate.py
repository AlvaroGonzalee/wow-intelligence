from pathlib import Path
import argparse
import hashlib
import json
import sys

from wow_intelligence.config.ollama import OllamaConfig
from wow_intelligence.ollama.client import generate
from wow_intelligence.ollama.embeddings import OllamaEmbedder
from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.rag.evaluation.benchmark import evaluate
from wow_intelligence.rag.evaluation.reference_questions import load_questions
from wow_intelligence.rag.generation.prompts import RAG_INSTRUCTIONS
from wow_intelligence.rag.pipeline import RagPipeline
from wow_intelligence.rag.storage.json_loader import load_knowledge_base


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
        pipeline = RagPipeline(load_knowledge_base(args.corpus), include_catalog=args.include_catalog,
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
