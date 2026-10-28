from datetime import datetime, timezone
from pathlib import Path
import argparse

from wow_intelligence.rag.ingestion.ahledger.common import ImportError
from wow_intelligence.rag.ingestion.ahledger.items import MAX_RESPONSE_BYTES, fetch_item, write_review_bundle


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
