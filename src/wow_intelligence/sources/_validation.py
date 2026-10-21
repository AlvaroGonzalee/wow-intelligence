#Validate shared source and evidence metadata without accessing the network.

from datetime import datetime
from urllib.parse import urlsplit


def _require_text(value: str, field: str) -> None:
    #Reject missing or whitespace-only metadata instead of inventing defaults
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text.")


def _require_url(value: str, field: str) -> None:
   #Check citation URL structure without visiting it or proving ownership
    _require_text(value, field)
    if any(character.isspace() for character in value):
        raise ValueError(f"{field} must not contain whitespace.")
    parts = urlsplit(value)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without credentials.")
    # Accessing port also validates malformed or out-of-range port values.
    _ = parts.port


def _require_timestamp(value: datetime, field: str) -> None:
    #Require timezone-aware timestamps so review dates remain unambiguous
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime.")
