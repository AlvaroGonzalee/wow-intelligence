from dataclasses import dataclass

from wow_intelligence.rag.models.evidence import Evidence
from wow_intelligence.rag.models.source import Source


@dataclass(frozen=True)
class KnowledgeBase:
    # Hold the sources and evidence loaded into memory for retrieval.
    sources: tuple[Source, ...]
    evidence: tuple[Evidence, ...]
