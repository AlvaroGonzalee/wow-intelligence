from dataclasses import dataclass

from wow_intelligence.rag.models.evidence import Evidence


@dataclass(frozen=True)
class RagAnswer:
    # Separate English answer text from citations rendered by the application.
    text: str
    evidence: tuple[Evidence, ...] = ()
    validation_text: str | None = None
    method: str = "model"

    def sources_text(self) -> str:
        if not self.evidence:
            return "No sources were cited in the last accepted answer."
        return "Sources:\n" + "\n".join(
            f"[{index}] {item.evidence_id} | {item.url} | Build: {item.game_build} | "
            f"Status: {item.status} | Verified: {(item.verified_at.isoformat() if item.verified_at else "not independently verified")} | Source: {item.source.name} ({item.source.url})"
            for index, item in enumerate(self.evidence, 1)
        )
