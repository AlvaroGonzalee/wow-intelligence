from dataclasses import dataclass

from wow_intelligence.ollama.messages import Message
from wow_intelligence.rag.models.evidence import Evidence


@dataclass(frozen=True)
class PreparedTurn:
    # Pair a temporary model request with exactly the evidence it can cite.
    messages: list[Message]
    evidence: tuple[Evidence, ...]
    notice: str | None = None
    ranked_ids: tuple[str, ...] = ()
