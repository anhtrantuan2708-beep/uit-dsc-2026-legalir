#!/usr/bin/env python3
"""Evaluate LegalIR predictions against warmup labels.

This is a local diagnostic evaluator. It reports macro averages over queries:
recall = relevant retrieved / relevant gold, precision = relevant retrieved /
retrieved. The official scorer remains the authority if its exact aggregation
differs.
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path, help="warmup/train JSON with answer IDs")
    parser.add_argument("predictions", type=Path, help="submission JSON with answer IDs")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Evaluate only the first k predicted IDs per question",
    )
    args = parser.parse_args()

    gold = load(args.gold)
    pred = load(args.predictions)
    recalls = []
    precisions = []
    hits = 0
    missing = 0

    for query_id, row in gold.items():
        gold_ids = {str(value) for value in row.get("answer", [])}
        pred_row = pred.get(query_id, {})
        predicted_values = pred_row.get("answer", [])
        if args.top_k is not None:
            predicted_values = predicted_values[: args.top_k]
        pred_ids = {str(value) for value in predicted_values}
        if query_id not in pred:
            missing += 1
        relevant = len(gold_ids & pred_ids)
        recalls.append(relevant / len(gold_ids) if gold_ids else 0.0)
        precisions.append(relevant / len(pred_ids) if pred_ids else 0.0)
        hits += int(bool(gold_ids & pred_ids))

    n = len(recalls) or 1
    print(f"queries: {len(recalls)}")
    if args.top_k is not None:
        print(f"evaluated top-k: {args.top_k}")
    print(f"missing predictions: {missing}")
    print(f"hit@k: {hits / n:.4f}")
    print(f"macro recall: {sum(recalls) / n:.4f}")
    print(f"macro precision: {sum(precisions) / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
