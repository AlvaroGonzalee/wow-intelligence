from pathlib import Path
import argparse

from wow_intelligence.config.ollama import OllamaConfig
from wow_intelligence.ollama.embeddings import OllamaEmbedder
from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.rag.indexing.vector_index import build_index
from wow_intelligence.rag.pipeline import RagPipeline
from wow_intelligence.rag.storage.json_loader import load_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a new local embedding index from eligible corpus evidence.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--game-build")
    parser.add_argument("--include-catalog", action="store_true")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists. Choose a new index path.")
        config = OllamaConfig.from_env()
        pipeline = RagPipeline(load_knowledge_base(args.corpus), args.game_build, include_catalog=args.include_catalog)
        embedder = OllamaEmbedder(args.embedding_model, base_url=config.base_url, timeout=config.timeout)
        build_index(args.output, pipeline.retriever.evidence, embedder,
                    progress=lambda completed, total: print(f"Indexed {completed}/{total}", flush=True))
    except (ValueError, OSError, OllamaError) as exc:
        parser.error(str(exc))
    print(f"Index ready: {args.output}")


if __name__ == "__main__":
    main()
