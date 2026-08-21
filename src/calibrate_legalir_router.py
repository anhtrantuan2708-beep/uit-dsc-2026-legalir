#!/usr/bin/env python3
"""Calibrate a conservative top-5 LegalIR router on one fixed labelled block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def route(
    base: dict[str, Any],
    alternate: dict[str, Any],
    scores: dict[str, Any],
    min_score: float,
    min_margin: float,
    keep_base: int,
) -> tuple[dict[str, dict[str, list[str]]], int]:
    result: dict[str, dict[str, list[str]]] = {}
    routed = 0
    for query_id, row in base.items():
        base_answer = [str(value) for value in row.get("answer", [])]
        ranked_scores = scores.get(query_id, [])
        valid = [item for item in ranked_scores if item.get("score") is not None]
        confidence = float(valid[0]["score"]) if valid else float("-inf")
        margin = confidence - float(valid[1]["score"]) if len(valid) > 1 else float("inf")
        sources: list[list[str]] = [base_answer]
        if confidence >= min_score and margin >= min_margin:
            sources = [
                base_answer[:keep_base],
                [str(value) for value in alternate.get(query_id, {}).get("answer", [])],
                base_answer,
            ]
            routed += 1
        answer: list[str] = []
        for source in sources:
            for source_id in source:
                if source_id not in answer:
                    answer.append(source_id)
                if len(answer) == 5:
                    break
            if len(answer) == 5:
                break
        result[query_id] = {"answer": answer}
    return result, routed


def metrics(gold: dict[str, Any], predicted: dict[str, Any]) -> tuple[float, float]:
    recalls, precisions = [], []
    for query_id, row in gold.items():
        relevant = {str(value) for value in row.get("answer", [])}
        selected = {str(value) for value in predicted.get(query_id, {}).get("answer", [])[:5]}
        hits = len(relevant & selected)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / len(selected) if selected else 0.0)
    return sum(recalls) / len(recalls), sum(precisions) / len(precisions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--keep-base", type=int, default=4)
    parser.add_argument("--min-scores", type=float, nargs="+", required=True)
    parser.add_argument("--min-margins", type=float, nargs="+", required=True)
    args = parser.parse_args()

    gold, base, alternate, scores = map(load, (args.gold, args.base, args.alternate, args.scores))
    base_recall, base_precision = metrics(gold, base)
    trials = []
    selected_prediction = base
    selected_trial = None
    for min_score in args.min_scores:
        for min_margin in args.min_margins:
            prediction, routed = route(base, alternate, scores, min_score, min_margin, args.keep_base)
            recall, precision = metrics(gold, prediction)
            trial = {
                "min_score": min_score,
                "min_margin": min_margin,
                "routed_queries": routed,
                "recall": round(recall, 6),
                "precision": round(precision, 6),
                "recall_delta_vs_base": round(recall - base_recall, 6),
                "precision_delta_vs_base": round(precision - base_precision, 6),
            }
            trials.append(trial)
            # Prefer Recall, then Precision, then fewer changes when tied.
            if selected_trial is None or (
                recall,
                precision,
                -routed,
            ) > (
                selected_trial["recall"],
                selected_trial["precision"],
                -selected_trial["routed_queries"],
            ):
                selected_prediction, selected_trial = prediction, trial

    assert selected_trial is not None
    report = {
        "base": {"recall": round(base_recall, 6), "precision": round(base_precision, 6)},
        "selected": selected_trial,
        "trials": trials,
    }
    atomic_dump(args.output, selected_prediction)
    atomic_dump(args.report, report)
    print(json.dumps({"base": report["base"], "selected": selected_trial}, indent=2))


if __name__ == "__main__":
    main()
