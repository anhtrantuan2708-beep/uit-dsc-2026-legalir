#!/usr/bin/env python3
"""Make a small deterministic dev subset for fast experiment smoke tests."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    selected_ids = sorted(rows, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[: args.size]
    subset = {query_id: rows[query_id] for query_id in selected_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(subset)} smoke-test queries to {args.output}")


if __name__ == "__main__":
    main()
