import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class OllamaConfig:
    # Hold connection settings shared by every request in one chat session.
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or any(
            character.isspace() for character in self.base_url
        ):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP or HTTPS URL without whitespace.")
        try:
            url = urlsplit(self.base_url)
            port = url.port
            valid_url = (
                url.scheme in ("http", "https") and url.hostname
                and url.username is None and url.password is None
                and not url.query and not url.fragment
                and (port is None or port > 0)
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("OLLAMA_BASE_URL must be an HTTP or HTTPS URL without credentials, query, or fragment.")
        if not isinstance(self.model, str) or not self.model.strip() or any(
            character.isspace() for character in self.model
        ):
            raise ValueError("OLLAMA_MODEL must be a non-empty model name without whitespace.")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("OLLAMA_TIMEOUT must be a finite number greater than zero.")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OllamaConfig":
        values = os.environ if environ is None else environ
        defaults = cls()
        try:
            timeout = float(values.get("OLLAMA_TIMEOUT", str(defaults.timeout)))
        except ValueError as exc:
            raise ValueError("OLLAMA_TIMEOUT must be a finite number greater than zero.") from exc
        return cls(
            base_url=values.get("OLLAMA_BASE_URL", defaults.base_url),
            model=values.get("OLLAMA_MODEL", defaults.model),
            timeout=timeout,
        )
