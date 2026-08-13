#!/usr/bin/env python3
"""Merge non-overlapping LegalIR ranking shards into one JSON object."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()

    merged = {}
    for path in args.shards:
        rows = json.loads(path.read_text(encoding="utf-8"))
        duplicate = set(merged) & set(rows)
        if duplicate:
            raise SystemExit(f"duplicate query IDs in {path}: {len(duplicate)}")
        merged.update(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(args.shards)} shards and {len(merged)} queries into {args.output}")


if __name__ == "__main__":
    main()
