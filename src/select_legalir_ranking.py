#!/usr/bin/env python3
"""Select an alternate LegalIR ranking for queries above a confidence threshold."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("probabilities", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    base = load(args.base)
    alternate = load(args.alternate)
    probabilities = load(args.probabilities)
    selected = 0
    result = {}
    for query_id, row in base.items():
        use_alternate = float(probabilities.get(query_id, 0.0)) >= args.threshold
        result[query_id] = alternate.get(query_id, row) if use_alternate else row
        selected += int(use_alternate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions; alternate selected for {selected} queries")


if __name__ == "__main__":
    main()
