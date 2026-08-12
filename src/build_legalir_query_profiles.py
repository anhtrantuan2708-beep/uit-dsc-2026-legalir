#!/usr/bin/env python3
"""Build document query profiles from labelled LegalIR training questions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path, help="Labelled train JSON")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    profiles: dict[str, list[str]] = defaultdict(list)
    for row in queries.values():
        question = row.get("question", "")
        for document_id in row.get("answer", []) or []:
            profiles[str(document_id)].append(question)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(profiles), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"profiled documents: {len(profiles)}")
    print(f"source questions: {len(queries)}")


if __name__ == "__main__":
    main()
