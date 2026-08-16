#!/usr/bin/env python3
"""Evaluate LegalIR predictions by whether gold documents appeared in train."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    gold, train, predictions = load(args.gold), load(args.labelled_train), load(args.predictions)
    seen = {str(value) for row in train.values() for value in row.get("answer", [])}
    stats = defaultdict(lambda: [0, 0.0, 0.0])

    for query_id, row in gold.items():
        gold_ids = {str(value) for value in row.get("answer", [])}
        predicted = {
            str(value)
            for value in predictions.get(query_id, {}).get("answer", [])[: args.top_k]
        }
        relevant = len(gold_ids & predicted)
        if gold_ids <= seen:
            group = "all_seen"
        elif gold_ids.isdisjoint(seen):
            group = "fully_unseen"
        else:
            group = "mixed"
        stats[group][0] += 1
        stats[group][1] += relevant / len(gold_ids) if gold_ids else 0.0
        stats[group][2] += relevant / len(predicted) if predicted else 0.0

    for group, (count, recall, precision) in sorted(stats.items()):
        print(f"{group:12s} n={count:4d} recall={recall / count:.4f} precision={precision / count:.4f}")


if __name__ == "__main__":
    main()
