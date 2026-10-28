from dataclasses import fields
from datetime import datetime
from pathlib import Path
import json
import re

from wow_intelligence.rag.models.evidence import Evidence
from wow_intelligence.rag.models.knowledge_base import KnowledgeBase
from wow_intelligence.rag.models.source import Source


MAX_JSON_BYTES = 5 * 1024 * 1024


class KnowledgeBaseLoadError(ValueError):
    # Report failures reading or validating a knowledge-base JSON file.
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise KnowledgeBaseLoadError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _record(value: object, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise KnowledgeBaseLoadError("Each record must be an object with supported fields only.")
    for key, item in value.items():
        if isinstance(item, str) and any(
            (ord(char) < 32 and not (key == "content" and char in "\n\t"))
            or 127 <= ord(char) <= 159 for char in item
        ):
            raise KnowledgeBaseLoadError("Corpus text must not contain terminal control characters.")
    return dict(value)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise KnowledgeBaseLoadError("Identifiers must contain 1-80 letters, digits, dots, underscores, or hyphens.")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise KnowledgeBaseLoadError("Timestamps must be ISO 8601 strings with a timezone.")
    return datetime.fromisoformat(value)


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise KnowledgeBaseLoadError("Corpus exceeds the 5 MiB limit.")
        root = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(root, dict) or set(root) != {"schema_version", "sources", "evidence"}:
            raise KnowledgeBaseLoadError("Corpus requires schema_version, sources, and evidence.")
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise KnowledgeBaseLoadError("Unsupported corpus schema version.")
        if not isinstance(root["sources"], list) or not isinstance(root["evidence"], list):
            raise KnowledgeBaseLoadError("sources and evidence must be arrays.")
        sources = {}
        for record in root["sources"]:
            values = _record(record, {field.name for field in fields(Source)})
            source_id = _identifier(values.get("source_id"))
            if source_id in sources:
                raise KnowledgeBaseLoadError(f"Duplicate source ID: {source_id}.")
            sources[source_id] = Source(**values)
        evidence = {}
        allowed = {field.name for field in fields(Evidence)} - {"source"} | {"source_id"}
        for record in root["evidence"]:
            values = _record(record, allowed)
            evidence_id = _identifier(values.get("evidence_id"))
            if evidence_id in evidence:
                raise KnowledgeBaseLoadError(f"Duplicate evidence ID: {evidence_id}.")
            source_id = _identifier(values.pop("source_id", None))
            if source_id not in sources:
                raise KnowledgeBaseLoadError(f"Unknown source ID: {source_id}.")
            values["source"] = sources[source_id]
            values["retrieved_at"] = _timestamp(values.get("retrieved_at"))
            if values.get("verified_at") is not None:
                values["verified_at"] = _timestamp(values["verified_at"])
            evidence[evidence_id] = Evidence(**values)
        return KnowledgeBase(tuple(sources.values()), tuple(evidence.values()))
    except KnowledgeBaseLoadError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise KnowledgeBaseLoadError(f"Could not load corpus: {exc}") from exc
