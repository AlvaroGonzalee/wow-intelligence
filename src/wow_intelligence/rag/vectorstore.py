from dataclasses import asdict
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import tempfile

from wow_intelligence.sources import Evidence
from .embedder import normalized_vector


def evidence_fingerprint(evidence: tuple[Evidence, ...]) -> str:
    records = [asdict(item) for item in sorted(evidence, key=lambda item: item.evidence_id)]
    raw = json.dumps(records, sort_keys=True, ensure_ascii=False, default=lambda value: value.isoformat()).encode()
    return hashlib.sha256(raw).hexdigest()


def build_index(path: Path, evidence: tuple[Evidence, ...], embedder, *, progress=None) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError("Index already exists. Choose a new path; snapshots are never overwritten.")
    if not evidence:
        raise ValueError("There is no eligible evidence to index.")
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("Index evidence IDs must be unique.")
    identity = embedder.identity
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vector-index-", dir=path.parent) as temporary:
        staged = Path(temporary) / "index.sqlite3"
        dimensions = None
        with closing(sqlite3.connect(staged)) as connection, connection:
            connection.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
            connection.execute("CREATE TABLE vectors (evidence_id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            for offset in range(0, len(evidence), 32):
                batch = evidence[offset:offset + 32]
                vectors = embedder.embed([item.content for item in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Embedding batch count does not match the evidence.")
                for item, vector in zip(batch, vectors):
                    vector = normalized_vector(vector, dimensions)
                    dimensions = len(vector)
                    connection.execute("INSERT INTO vectors VALUES (?, ?)", (item.evidence_id, struct.pack(f"<{dimensions}f", *vector)))
                if progress:
                    progress(min(offset + 32, len(evidence)), len(evidence))
            metadata = {"schema_version": 1, "evidence_sha256": evidence_fingerprint(evidence),
                        "embedding": identity, "dimensions": dimensions, "records": len(evidence)}
            connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        os.link(staged, path)


class VectorIndex:
    # Bind read-only vectors to the exact eligible evidence and model digest.
    def __init__(self, path: Path, evidence: tuple[Evidence, ...], identity: dict):
        path = Path(path).resolve()
        if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("Vector index is missing or exceeds 512 MiB.")
        try:
            connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                connection.execute("PRAGMA query_only = ON")
                metadata_rows = connection.execute("SELECT value FROM metadata").fetchall()
                if len(metadata_rows) != 1:
                    raise ValueError("Invalid vector index metadata.")
                metadata = json.loads(metadata_rows[0][0])
                if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
                    raise ValueError("Unsupported vector index schema.")
                if metadata.get("evidence_sha256") != evidence_fingerprint(evidence):
                    raise ValueError("Vector index is stale or uses a different build or eligibility policy. Rebuild it.")
                if metadata.get("embedding") != identity:
                    raise ValueError("Vector index embedding model or digest changed. Rebuild it.")
                dimensions = metadata.get("dimensions")
                if type(dimensions) is not int or not 1 <= dimensions <= 16384:
                    raise ValueError("Invalid vector dimensions in index.")
                rows = connection.execute("SELECT evidence_id, vector FROM vectors ORDER BY evidence_id").fetchall()
            finally:
                connection.close()
            if type(metadata.get("records")) is not int or metadata["records"] != len(evidence) or len(rows) != len(evidence) or {key for key, _ in rows} != {item.evidence_id for item in evidence}:
                raise ValueError("Vector index evidence IDs do not match the corpus.")
            self.dimensions = dimensions
            self.vectors = {}
            for key, raw in rows:
                if not isinstance(raw, bytes) or len(raw) != dimensions * 4:
                    raise ValueError("Invalid stored vector size.")
                self.vectors[key] = normalized_vector(struct.unpack(f"<{dimensions}f", raw), dimensions)
        except (sqlite3.Error, UnicodeError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read vector index: {exc}") from exc

    def search(self, vector, limit: int = 12) -> tuple[str, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("Vector search limit must be a positive integer.")
        query = normalized_vector(vector, self.dimensions)
        scores = [(sum(left * right for left, right in zip(query, vector)), key) for key, vector in self.vectors.items()]
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return tuple(key for score, key in scores[:limit] if score > 0)
