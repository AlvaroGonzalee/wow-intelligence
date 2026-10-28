import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wow_intelligence.config import OllamaConfig
from wow_intelligence.ollama import OllamaError


def normalized_vector(values, dimensions: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16384:
        raise ValueError("Embedding must contain 1-16384 numeric dimensions.")
    if dimensions is not None and len(values) != dimensions:
        raise ValueError("Embedding dimensions do not match the index.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Embedding contains invalid numeric values.")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite nonzero norm.")
    return tuple(value / norm for value in values)


class OllamaEmbedder:
    # Request bounded embeddings without truncation, retries or model downloads.
    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:11434", timeout: float = 120):
        self.config = OllamaConfig(base_url=base_url, model=model, timeout=timeout)
        self.model = model
        self._identity = None

    def _request(self, path: str, payload: dict | None = None) -> dict:
        request = Request(
            self.config.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("Embedding response exceeds 16 MiB.")
            result = json.loads(raw)
            if not isinstance(result, dict) or "error" in result:
                raise ValueError("Invalid embedding response.")
            return result
        except HTTPError as exc:
            exc.close()
            raise OllamaError(f"Ollama embeddings returned HTTP {exc.code}. Check that the selected embedding model is installed and supports embeddings.") from exc
        except (URLError, TimeoutError) as exc:
            raise OllamaError("Could not reach the Ollama embedding service. No lexical fallback was used.") from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError(f"Invalid Ollama embedding response: {exc}") from exc

    @property
    def identity(self) -> dict:
        if self._identity is None:
            models = self._request("/api/tags").get("models")
            if not isinstance(models, list):
                raise OllamaError("Ollama did not return its installed model list.")
            names = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
            model = next((item for item in models if isinstance(item, dict) and item.get("name") in names), None)
            if model is None or not isinstance(model.get("digest"), str) or not model["digest"]:
                raise OllamaError(f"Embedding model {self.model!r} is not installed or has no digest. Install it explicitly before indexing.")
            if "capabilities" in model:
                if not isinstance(model["capabilities"], list) or "embedding" not in model["capabilities"]:
                    raise OllamaError(f"Model {self.model!r} does not advertise embedding support. Select an embedding model.")
            self._identity = {"model": model["name"], "digest": model["digest"]}
        return dict(self._identity)

    def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        if not isinstance(texts, list) or not 1 <= len(texts) <= 32 or any(
            not isinstance(text, str) or not text.strip() or len(text.encode()) > 65536 for text in texts
        ):
            raise ValueError("Embed 1-32 nonempty texts of at most 64 KiB each.")
        self.identity
        result = self._request("/api/embed", {"model": self.model, "input": texts, "truncate": False})
        vectors = result.get("embeddings")
        try:
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError("Embedding count does not match the request.")
            normalized = tuple(normalized_vector(vector) for vector in vectors)
            if len({len(vector) for vector in normalized}) != 1:
                raise ValueError("Embedding response has inconsistent dimensions.")
            return normalized
        except ValueError as exc:
            raise OllamaError(str(exc)) from exc
