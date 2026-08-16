#!/usr/bin/env python3
"""Compare agreement patterns between paired Dev and Public ranking sources."""

import argparse
import itertools
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def agreement(left: dict, right: dict, top_k: int) -> tuple[float, float]:
    query_ids = set(left) & set(right)
    top1 = 0
    overlap = 0.0
    for query_id in query_ids:
        left_ids = [str(value) for value in left[query_id].get("answer", [])[:top_k]]
        right_ids = [str(value) for value in right[query_id].get("answer", [])[:top_k]]
        top1 += int(bool(left_ids and right_ids and left_ids[0] == right_ids[0]))
        overlap += len(set(left_ids) & set(right_ids)) / top_k
    count = len(query_ids) or 1
    return top1 / count, overlap / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", action="append", type=Path, required=True)
    parser.add_argument("--public-source", action="append", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if len(args.dev_source) != len(args.public_source):
        raise ValueError("Dev and Public source counts must match")

    dev = [load(path) for path in args.dev_source]
    public = [load(path) for path in args.public_source]
    print("source lengths: dev -> public")
    for index, (dev_rows, public_rows) in enumerate(zip(dev, public), start=1):
        dev_length = sum(len(row.get("answer", [])) for row in dev_rows.values()) / len(dev_rows)
        public_length = sum(len(row.get("answer", [])) for row in public_rows.values()) / len(public_rows)
        print(f"s{index}: {dev_length:.1f} -> {public_length:.1f}")

    shifts = []
    for left, right in itertools.combinations(range(len(dev)), 2):
        dev_top1, dev_overlap = agreement(dev[left], dev[right], args.top_k)
        public_top1, public_overlap = agreement(public[left], public[right], args.top_k)
        shifts.append(
            (
                abs(dev_overlap - public_overlap),
                left + 1,
                right + 1,
                dev_top1,
                public_top1,
                dev_overlap,
                public_overlap,
            )
        )
    print("largest pairwise agreement shifts:")
    for _, left, right, dev_top1, public_top1, dev_overlap, public_overlap in sorted(shifts, reverse=True)[:10]:
        print(
            f"s{left}/s{right}: top1 {dev_top1:.3f}->{public_top1:.3f}; "
            f"overlap@{args.top_k} {dev_overlap:.3f}->{public_overlap:.3f}"
        )


if __name__ == "__main__":
    main()
