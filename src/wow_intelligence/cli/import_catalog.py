from datetime import datetime
from pathlib import Path
import argparse

from wow_intelligence.rag.ingestion.ahledger.catalog import import_export


def main() -> None:
    parser = argparse.ArgumentParser(description='Import a saved AHledger catalog export into a new local snapshot.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retrieved-at', required=True, help='Original timezone-aware retrieval timestamp.')
    args = parser.parse_args()
    try:
        path = import_export(args.input, args.output, datetime.fromisoformat(args.retrieved_at))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Catalog snapshot written to {path}. Use --include-catalog to query source-reported records.')


if __name__ == "__main__":
    main()
