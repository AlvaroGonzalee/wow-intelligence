import argparse
from importlib.metadata import version

from wow_intelligence.chat import run_chat
from wow_intelligence.config import OllamaConfig
from wow_intelligence.ollama import OllamaError
from wow_intelligence.rag import RagPipeline, load_corpus
from wow_intelligence.rag.embedder import OllamaEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wow-intelligence",
        description="Experimental terminal chat using local Ollama.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"WoW Intelligence {version('wow-intelligence')}",
    )
    parser.add_argument("--corpus", metavar="PATH", help="Load a reviewed local JSON corpus and enable RAG.")
    parser.add_argument("--game-build", metavar="BUILD", help="Override the newest numeric build recorded in the corpus.")
    parser.add_argument("--check-corpus", action="store_true", help="Validate the corpus and report eligible evidence, then exit.")
    parser.add_argument("--include-catalog", action="store_true", help="Allow explicitly labelled source catalog facts; these are not independently verified in game.")
    parser.add_argument("--index", help="Use a compatible local vector index for hybrid retrieval.")
    parser.add_argument("--embedding-model", help="Installed Ollama embedding model used to build the index.")
    args = parser.parse_args()
    if bool(args.index) != bool(args.embedding_model) or (args.index and not args.corpus):
        parser.error("--index and --embedding-model must be supplied together with --corpus.")
    if args.include_catalog and not args.corpus:
        parser.error("--include-catalog requires --corpus.")
    if args.game_build and not args.corpus:
        parser.error("--game-build requires --corpus.")
    if args.check_corpus and not args.corpus:
        parser.error("--check-corpus requires --corpus.")
    try:
        embedder = None
        if args.index:
            embedding_config = OllamaConfig.from_env()
            embedder = OllamaEmbedder(args.embedding_model, base_url=embedding_config.base_url, timeout=embedding_config.timeout)
        rag = RagPipeline(load_corpus(args.corpus), args.game_build, include_catalog=args.include_catalog,
                          index_path=args.index, embedder=embedder) if args.corpus else None
        if args.check_corpus:
            print(f"Corpus is valid. Eligible evidence for build {rag.game_build}: {len(rag.retriever.evidence)}.")
            print(rag.coverage_text())
            print("Metadata validation does not independently verify facts or permissions.")
            return
        config = OllamaConfig.from_env()
    except (ValueError, OllamaError) as exc:
        parser.error(str(exc))
    if rag is None:
        run_chat(config)
    else:
        run_chat(config, rag=rag)


if __name__ == "__main__":
    main()
