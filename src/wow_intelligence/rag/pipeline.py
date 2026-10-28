from collections import Counter
from pathlib import Path
import re

from wow_intelligence.chat.policies.conversation import ConversationPolicy
from wow_intelligence.ollama.messages import Message
from wow_intelligence.rag.generation.answer_parser import parse_answer
from wow_intelligence.rag.generation.catalog_answers import catalog_answer, catalog_names
from wow_intelligence.rag.generation.context_builder import prepare_context
from wow_intelligence.rag.generation.prompts import ABSTENTION
from wow_intelligence.rag.generation.schema import answer_schema
from wow_intelligence.rag.models.knowledge_base import KnowledgeBase
from wow_intelligence.rag.models.prepared_turn import PreparedTurn
from wow_intelligence.rag.models.rag_answer import RagAnswer
from wow_intelligence.rag.retrieval.bm25 import BM25Retriever


class RagPipeline:
    # Coordinate retrieval, context preparation and cited answers for one build.
    def __init__(self, knowledge_base: KnowledgeBase, game_build: str | None = None, *, include_catalog: bool = False,
                 index_path: Path | None = None, embedder=None) -> None:
        if game_build is None:
            game_build = self.select_build(knowledge_base)
        self.game_build = game_build
        self.knowledge_base = knowledge_base
        self.include_catalog = include_catalog
        self.retriever = BM25Retriever(knowledge_base, game_build, include_catalog=include_catalog)
        self.language_names = catalog_names(self.retriever.evidence)
        self.retrieval_mode = "bm25"
        if (index_path is None) != (embedder is None):
            raise ValueError("Hybrid retrieval requires both an index and an embedding model.")
        if index_path is not None:
            from wow_intelligence.rag.retrieval.hybrid import HybridRetriever
            from wow_intelligence.rag.indexing.vector_index import VectorIndex
            index = VectorIndex(index_path, self.retriever.evidence, embedder.identity)
            self.retriever = HybridRetriever(self.retriever, index, embedder)
            self.retrieval_mode = "hybrid"

    @staticmethod
    def select_build(knowledge_base: KnowledgeBase) -> str:
        builds = {item.game_build for item in knowledge_base.evidence
                  if item.game_build and item.game_build.strip().casefold() != "unknown"}
        if len(builds) == 1:
            return next(iter(builds))
        if builds and all(re.fullmatch(r"\d+(?:\.\d+)*", build) for build in builds):
            parts = {tuple(map(int, build.split("."))): build for build in builds}
            if len(parts) == len(builds) and len({len(key) for key in parts}) == 1:
                return parts[max(parts)]
        raise ValueError("The corpus has no unambiguous target build. Review its metadata or supply --game-build.")

    def prepare(self, history: list[Message], query: str) -> PreparedTurn:
        other_edition = re.search(r"\b(?:retail|classic|tbc|wrath)\b", query, re.IGNORECASE)
        builds = re.findall(r"\b(?:build|patch)\s+(\d+(?:\.\d+)*)", query, re.IGNORECASE)
        incompatible_build = any(not (
            build == self.game_build or self.game_build.startswith(build + ".")
            or ("." not in build and build == self.game_build.rsplit(".", 1)[-1])
        ) for build in builds)
        if other_edition or incompatible_build:
            return PreparedTurn([], (), f"This corpus covers Forever build {self.game_build}. It does not establish facts for the requested edition or build.")
        search_query = query
        if ConversationPolicy.needs_context(query):

            previous = next((message["content"] for message in reversed(history)
                             if message["role"] == "user"
                             and not ConversationPolicy.needs_context(message["content"])), "")
            search_query = f"{previous} {query}"
        return prepare_context(history, query, self.retriever.search(search_query))

    def answer(self, prepared: PreparedTurn, query: str, *, generator) -> RagAnswer:
        if not prepared.evidence:
            return RagAnswer(prepared.notice or ABSTENTION, method="abstention")
        structured = catalog_answer(query, prepared.evidence)
        if structured is not None:
            return structured
        return parse_answer(generator(prepared.messages, response_schema=answer_schema(prepared.evidence)), prepared.evidence)

    def coverage_text(self) -> str:
        builds = Counter(item.game_build or "unknown" for item in self.knowledge_base.evidence)
        statuses = Counter(item.status for item in self.knowledge_base.evidence)
        return "\n".join([
            f"Selected Forever build: {self.game_build}",
            f"Retrieval: {self.retrieval_mode}.",
            "Source catalog mode: " + ("enabled; not independently verified in game." if self.include_catalog else "disabled."),
            f"Loaded sources: {len(self.knowledge_base.sources)}; evidence records: {len(self.knowledge_base.evidence)}.",
            f"Eligible evidence: {len(self.retriever.evidence)}.",
            "Builds: " + (", ".join(f"{key}: {value}" for key, value in sorted(builds.items())) or "none"),
            "Review states: " + (", ".join(f"{key}: {value}" for key, value in sorted(statuses.items())) or "none"),
            "These counts describe this local corpus, not complete coverage of the game.",
        ])
