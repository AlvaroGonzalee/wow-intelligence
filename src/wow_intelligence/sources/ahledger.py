import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .source import Source
from .evidence import Evidence
from ._validation import _require_text, _require_url
from wow_intelligence.rag.corpus import _unique_object, load_corpus


MAX_RESPONSE_BYTES = 1024 * 1024


class ImportError(ValueError):
    # Describe an invalid API response or unsafe import destination.
    pass


def _ahledger_url(value: object) -> str:
    _require_url(value, "AHledger URL")
    if urlsplit(value).scheme != "https" or urlsplit(value).hostname != "ahledger.com":
        raise ImportError("Item and attribution links must point to https://ahledger.com.")
    return value


def item_corpus(payload: dict, retrieved_at: datetime) -> dict:
    if not isinstance(payload, dict) or type(payload.get("id")) is not int or payload["id"] <= 0:
        raise ImportError("An AHledger item must have a positive integer id.")
    for key in ("name", "quality", "itemClass", "itemSubclass"):
        _require_text(payload.get(key), key)
    for key in ("level", "requiredLevel", "vendorSell"):
        if key not in payload or (payload[key] is not None and (type(payload[key]) is not int or payload[key] < 0)):
            raise ImportError(f"{key} must be a non-negative integer or null.")
    attribution = payload.get("attribution")
    if not isinstance(attribution, dict):
        raise ImportError("The response must include attribution.")
    _require_text(attribution.get("text"), "attribution text")
    source = Source(
        source_id="ahledger-items", name=attribution["text"],
        url=_ahledger_url(attribution.get("url")), game_version="forever",
        permission_status="approved", enabled=True,
        permission_reference="https://ahledger.com/terms ; API reuse with attribution reviewed 2026-10-21.",
    )
    facts = {
        "item ID": payload["id"], "name": payload["name"], "quality": payload["quality"],
        "class": payload["itemClass"], "subclass": payload["itemSubclass"],
        "item level": payload["level"], "required level": payload["requiredLevel"],
        "vendor sell value in copper": payload["vendorSell"],
    }
    content = "AHledger reports " + "; ".join(
        f"{key}: {value if value is not None else 'unknown'}" for key, value in facts.items()
    ) + ". The response does not identify a game build; applicability to Forever is unverified."
    evidence = Evidence(
        evidence_id=f"ahledger-item-{payload['id']}", source=source,
        url=_ahledger_url(payload.get("url")), content=content,
        game_version="forever", retrieved_at=retrieved_at, status="unverified_for_forever",
    )
    record = asdict(evidence)
    record.pop("source")
    record["source_id"] = source.source_id
    record["retrieved_at"] = retrieved_at.isoformat()
    return {"schema_version": 1, "sources": [asdict(source)], "evidence": [record]}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ImportError("API redirects are not followed. Review the new endpoint first.")


def fetch_item(item_id: int) -> bytes:
    if type(item_id) is not int or item_id < 1:
        raise ImportError("Item ID must be a positive integer.")
    url = f"https://api.ahledger.com/v1/items/{item_id}?game=forever"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "WoWIntelligence/0.0.1"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        retry_after = exc.headers.get("Retry-After", "not supplied") if exc.headers else "not supplied"
        exc.close()
        raise ImportError(f"AHledger returned HTTP {exc.code}; Retry-After: {retry_after}. No retry was made.") from exc
    except (URLError, TimeoutError) as exc:
        raise ImportError("AHledger request failed. Do not retry before any server Retry-After interval.") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ImportError("API response exceeds 1 MiB.")
    return raw


def write_review_bundle(raw: bytes, output: Path, retrieved_at: datetime, *, expected_id: int | None = None) -> Path:
    output = Path(output)
    if output.exists():
        raise ImportError("Output already exists. Choose a new directory; prior imports are preserved.")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ImportError("API response exceeds 1 MiB.")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        corpus = item_corpus(payload, retrieved_at)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ImportError(f"Invalid item response: {exc}") from exc
    if expected_id is not None and payload["id"] != expected_id:
        raise ImportError("The response item ID does not match the requested item.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ahledger-", dir=output.parent) as temporary:
        folder = Path(temporary)
        (folder / "raw.json").write_bytes(raw)
        (folder / "corpus.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        review = {
            "source_url": f"https://api.ahledger.com/v1/items/{payload['id']}?game=forever",
            "retrieved_at": retrieved_at.isoformat(), "sha256": hashlib.sha256(raw).hexdigest(),
            "permission_reference": "https://ahledger.com/terms",
            "attribution": payload["attribution"],
            "review_status": "pending", "build": None,
            "limitations": ["No build in API response.", "No independent fact verification.",
                            "Null fields remain unknown, not zero.", "No drop locations or rates inferred."],
        }
        (folder / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        load_corpus(folder / "corpus.json")
        if output.exists():
            raise ImportError("Output already exists.")
        os.rename(folder, output)
    return output / "corpus.json"


def main() -> None:
    parser = argparse.ArgumentParser(description='Import an AHledger item API response into a local review bundle.')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--item", type=int, help="Fetch one item from the public API.")
    choice.add_argument("--input", type=Path, help="Import a previously saved API response.")
    parser.add_argument("--output", type=Path, required=True, help="New local review-bundle directory.")
    parser.add_argument("--retrieved-at", help="Original timezone-aware ISO timestamp, required with --input.")
    args = parser.parse_args()
    if bool(args.input) != bool(args.retrieved_at):
        parser.error("--input and --retrieved-at must be supplied together.")
    try:
        if args.output.exists():
            raise ImportError("Output already exists. No request was made.")
        if args.input:
            with args.input.open("rb") as stream:
                raw = stream.read(MAX_RESPONSE_BYTES + 1)
        else:
            raw = fetch_item(args.item)
        retrieved_at = datetime.fromisoformat(args.retrieved_at) if args.input else datetime.now(timezone.utc)
        path = write_review_bundle(raw, args.output, retrieved_at, expected_id=args.item)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Review corpus written to {path}.")
    print("Evidence remains unverified with no build and cannot be used to answer game questions.")


if __name__ == "__main__":
    main()
