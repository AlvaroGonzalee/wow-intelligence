from datetime import datetime
from urllib.parse import urlsplit


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text.")


def _require_url(value: str, field: str) -> None:
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

    _ = parts.port


def _require_timestamp(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime.")
