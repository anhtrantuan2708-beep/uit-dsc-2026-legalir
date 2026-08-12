#!/usr/bin/env python3
"""Print a compact summary of the LegalIR warmup JSON."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Warmup must be a JSON object keyed by query ID")

    answer_counts = Counter(len(item.get("answer", [])) for item in data.values())
    print(f"queries: {len(data)}")
    print(f"answer_count_distribution: {dict(sorted(answer_counts.items()))}")
    print("sample:")
    query_id, sample = next(iter(data.items()))
    print(json.dumps({query_id: sample}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
