from dataclasses import dataclass
from typing import Literal

from ._validation import _require_text, _require_url


PermissionStatus = Literal["pending_review", "approved", "rejected"]


@dataclass(frozen=True)
class Source:
    # Identify a source and record an explicit decision about permitted use.
    source_id: str
    name: str
    url: str
    game_version: str
    permission_status: PermissionStatus = "pending_review"
    permission_reference: str | None = None
    enabled: bool = False

    def __post_init__(self) -> None:
        for field in ("source_id", "name", "game_version"):
            _require_text(getattr(self, field), field)
        _require_url(self.url, "url")
        if self.permission_status not in ("pending_review", "approved", "rejected"):
            raise ValueError("Unknown source permission status.")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if self.permission_reference is not None:
            _require_text(self.permission_reference, "permission_reference")
        if self.permission_status == "approved":
            _require_text(self.permission_reference, "permission_reference")
        if self.enabled and self.permission_status != "approved":
            raise ValueError("Only an approved source can be enabled.")
