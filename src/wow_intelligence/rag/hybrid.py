from collections import defaultdict

from wow_intelligence.ollama import OllamaError
from .vectorstore import VectorIndex


class HybridRetriever:
    # Fuse lexical and semantic ranks after applying the same evidence filters.
    def __init__(self, lexical, index: VectorIndex, embedder):
        self.lexical = lexical
        self.evidence = lexical.evidence
        self.index = index
        self.embedder = embedder
        self._records = {item.evidence_id: item for item in self.evidence}

    def search(self, query: str, limit: int = 12):
        if not isinstance(query, str) or type(limit) is not int or limit < 1:
            raise ValueError("Search requires text and a positive integer limit.")
        if not query.strip():
            return ()
        lexical = [item.evidence_id for item in self.lexical.search(query, limit=limit)]
        try:
            semantic = self.index.search(self.embedder.embed([query])[0], limit=limit)
        except ValueError as exc:
            raise OllamaError(f"Semantic retrieval failed: {exc} Rebuild the index if the model changed.") from exc
        scores = defaultdict(float)
        for ranking in (lexical, semantic):
            for rank, key in enumerate(ranking, 1):
                scores[key] += 1 / (60 + rank)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        return tuple(self._records[key] for key in ordered[:limit])
