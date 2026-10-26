from collections import Counter
import re

from wow_intelligence.ollama import Message
from wow_intelligence.conversation import ConversationPolicy
from .context import PreparedTurn, prepare_context
from .corpus import Corpus
from .retriever import Retriever


class RagPipeline:
    # Own an immutable corpus snapshot and a build-specific lexical index.
    def __init__(self, corpus: Corpus, game_build: str | None = None, *, include_catalog: bool = False) -> None:
        if game_build is None:
            game_build = self.select_build(corpus)
        self.game_build = game_build
        self.corpus = corpus
        self.include_catalog = include_catalog
        self.retriever = Retriever(corpus, game_build, include_catalog=include_catalog)

    @staticmethod
    def select_build(corpus: Corpus) -> str:
        builds = {item.game_build for item in corpus.evidence
                  if item.game_build and item.game_build.strip().casefold() != "unknown"}
        if len(builds) == 1:
            return next(iter(builds))
        if builds and all(re.fullmatch(r"\d+(?:\.\d+)*", build) for build in builds):
            parts = {tuple(map(int, build.split("."))): build for build in builds}
            if len(parts) == len(builds) and len({len(key) for key in parts}) == 1:
                return parts[max(parts)]
        raise ValueError("The corpus has no unambiguous target build. Review its metadata or supply --game-build.")

    def prepare(self, history: list[Message], query: str) -> PreparedTurn:
        search_query = query
        if ConversationPolicy.needs_context(query):

            previous = next((message["content"] for message in reversed(history)
                             if message["role"] == "user"
                             and not ConversationPolicy.needs_context(message["content"])), "")
            search_query = f"{previous} {query}"
        return prepare_context(history, query, self.retriever.search(search_query))

    def coverage_text(self) -> str:
        builds = Counter(item.game_build or "unknown" for item in self.corpus.evidence)
        statuses = Counter(item.status for item in self.corpus.evidence)
        return "\n".join([
            f"Selected Forever build: {self.game_build}",
            "Source catalog mode: " + ("enabled; not independently verified in game." if self.include_catalog else "disabled."),
            f"Loaded sources: {len(self.corpus.sources)}; evidence records: {len(self.corpus.evidence)}.",
            f"Eligible evidence: {len(self.retriever.evidence)}.",
            "Builds: " + (", ".join(f"{key}: {value}" for key, value in sorted(builds.items())) or "none"),
            "Review states: " + (", ".join(f"{key}: {value}" for key, value in sorted(statuses.items())) or "none"),
            "These counts describe this local corpus, not complete coverage of the game.",
        ])
