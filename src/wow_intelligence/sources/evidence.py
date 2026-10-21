"""Define versioned evidence and determine whether its metadata permits use."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ._validation import _require_text, _require_timestamp, _require_url
from .source import Source


EvidenceStatus = Literal[
    "unknown", "unverified_for_forever", "verified_for_forever", "superseded", "withheld",
]


@dataclass(frozen=True)
class Evidence:
    """Keep a claim or excerpt with its citation, game context, and review dates.

    Unknown or unverified records may omit a build and verification date. They
    can be represented for review but cannot support a verified answer. A source
    permission decision is separate from the review of an individual claim.
    """

    evidence_id: str
    source: Source
    url: str
    content: str
    game_version: str
    retrieved_at: datetime
    game_build: str | None = None
    status: EvidenceStatus = "unknown"
    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        #Validate provenance and require explicit context for verified records
        for field in ("evidence_id", "content", "game_version"):
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
            "unknown", "unverified_for_forever", "verified_for_forever", "superseded", "withheld",
        ):
            raise ValueError("Unknown evidence status.")
        if self.status == "verified_for_forever":
            if self.game_version != "forever":
                raise ValueError("Verified Forever evidence must refer to Forever.")
            if self.game_build is None or self.game_build.strip().casefold() == "unknown":
                raise ValueError("Verified evidence requires a known game build.")
            if self.verified_at is None:
                raise ValueError("Verified evidence requires a verification timestamp.")

    def is_usable_for(self, game_version: str, game_build: str) -> bool:
        """Check recorded eligibility for a query without silently mixing builds.

        This is a metadata gate, not proof of truth or freshness. Pending,
        disabled, superseded, and unverified records remain ineligible. Exact
        build matching is deliberately conservative until validity ranges exist.
        """
        return (
            self.source.permission_status == "approved"
            and self.source.enabled
            and self.status == "verified_for_forever"
            and self.game_version == "forever"
            and self.game_version == game_version
            and self.game_build == game_build
        )
