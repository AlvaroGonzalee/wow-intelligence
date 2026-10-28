from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from wow_intelligence.rag.models.source import Source
from wow_intelligence.rag.validation.metadata import _require_text, _require_timestamp, _require_url


EvidenceStatus = Literal[
    "unknown", "unverified_for_forever", "verified_for_forever", "source_reported", "superseded", "withheld",
]


@dataclass(frozen=True)
class Evidence:
    # Keep a claim or excerpt with its citation, game context, and review dates.
    evidence_id: str
    source: Source
    url: str
    content: str
    game_version: str
    retrieved_at: datetime
    game_build: str | None = None
    status: EvidenceStatus = "unknown"
    verified_at: datetime | None = None
    language: str = "en"

    def __post_init__(self) -> None:
        for field in ("evidence_id", "content", "game_version", "language"):
            _require_text(getattr(self, field), field)
        if not isinstance(self.source, Source):
            raise ValueError("Evidence must reference a Source object.")
        if self.game_version != self.source.game_version:
            raise ValueError("Evidence and source must have the same game version.")
        _require_url(self.url, "url")
        _require_timestamp(self.retrieved_at, "retrieved_at")
        if self.game_build is not None:
            _require_text(self.game_build, "game_build")
        if self.verified_at is not None:
            _require_timestamp(self.verified_at, "verified_at")
        if self.status not in (
            "unknown", "unverified_for_forever", "verified_for_forever", "source_reported", "superseded", "withheld",
        ):
            raise ValueError("Unknown evidence status.")
        if self.status in ("verified_for_forever", "source_reported"):
            if self.game_version != "forever":
                raise ValueError("Verified Forever evidence must refer to Forever.")
            if self.game_build is None or self.game_build.strip().casefold() == "unknown":
                raise ValueError("Verified evidence requires a known game build.")
            if self.status == "verified_for_forever" and self.verified_at is None:
                raise ValueError("Verified evidence requires a verification timestamp.")

    def is_usable_for(self, game_version: str, game_build: str, *, include_catalog: bool = False) -> bool:
        return (
            self.source.permission_status == "approved"
            and self.source.enabled
            and (self.status == "verified_for_forever" or (include_catalog and self.status == "source_reported"))
            and self.game_version == "forever"
            and self.game_version == game_version
            and self.language == "en"
            and self.game_build == game_build
        )
