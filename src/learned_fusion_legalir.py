#!/usr/bin/env python3
"""Learn a candidate-level LegalIR fusion model from existing rankings."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def document_frequency(labelled: dict) -> Counter[str]:
    return Counter(
        str(document_id)
        for row in labelled.values()
        for document_id in row.get("answer", [])
    )


def build_rows(
    query_ids: list[str],
    rankings: list[dict],
    frequency: Counter[str],
    gold: dict | None = None,
    include_frequency: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    row_keys: list[tuple[str, str]] = []

    for query_id in query_ids:
        positions = []
        candidates: set[str] = set()
        for ranking in rankings:
            answer = [str(value) for value in ranking.get(query_id, {}).get("answer", [])]
            position = {document_id: rank for rank, document_id in enumerate(answer, start=1)}
            positions.append(position)
            candidates.update(answer)

        gold_ids = {str(value) for value in gold.get(query_id, {}).get("answer", [])} if gold else set()
        for document_id in candidates:
            row: list[float] = []
            present = 0
            reciprocal_sum = 0.0
            best_rank = 101
            for position in positions:
                rank = position.get(document_id)
                if rank is None:
                    row.extend((0.0, 0.0, 0.0))
                    continue
                present += 1
                reciprocal = 1.0 / rank
                reciprocal_sum += reciprocal
                best_rank = min(best_rank, rank)
                row.extend((1.0, reciprocal, math.log1p(rank) / math.log(101)))
            row.extend((present / len(rankings), reciprocal_sum, 1.0 / best_rank))
            if include_frequency:
                row.append(math.log1p(frequency[document_id]))
            features.append(row)
            labels.append(int(document_id in gold_ids))
            groups.append(query_id)
            row_keys.append((query_id, document_id))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups),
        row_keys,
    )


def new_model(args: argparse.Namespace) -> HistGradientBoostingClassifier:
    class_weight = (
        "balanced"
        if args.positive_weight is None
        else {0: 1.0, 1: args.positive_weight}
    )
    return HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        class_weight=class_weight,
        random_state=42,
    )


def rankings_from_scores(
    row_keys: list[tuple[str, str]], scores: np.ndarray, top_k: int
) -> dict:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for (query_id, document_id), score in zip(row_keys, scores):
        grouped.setdefault(query_id, []).append((float(score), document_id))
    return {
        query_id: {
            "answer": [
                document_id
                for _, document_id in sorted(values, key=lambda item: (-item[0], item[1]))[:top_k]
            ]
        }
        for query_id, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-source", action="append", type=Path, required=True)
    parser.add_argument("--test-source", action="append", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--disable-frequency", action="store_true")
    args = parser.parse_args()

    gold = load(args.gold)
    frequency = document_frequency(load(args.labelled_train))
    train_rankings = [load(path) for path in args.train_source]
    query_ids = list(gold)
    x, y, groups, row_keys = build_rows(
        query_ids,
        train_rankings,
        frequency,
        gold,
        include_frequency=not args.disable_frequency,
    )
    print(f"train rows={len(y)} positives={int(y.sum())} features={x.shape[1]}")

    if args.test_source:
        if len(args.test_source) != len(args.train_source):
            raise ValueError("train-source and test-source counts must match")
        model = new_model(args)
        model.fit(x, y)
        test_rankings = [load(path) for path in args.test_source]
        test_ids = list(test_rankings[0])
        test_x, _, _, test_keys = build_rows(
            test_ids,
            test_rankings,
            frequency,
            include_frequency=not args.disable_frequency,
        )
        scores = model.predict_proba(test_x)[:, 1]
        result = rankings_from_scores(test_keys, scores, args.top_k)
    else:
        scores = np.zeros(len(y), dtype=np.float64)
        splitter = GroupKFold(n_splits=args.folds)
        for fold, (train_indices, valid_indices) in enumerate(splitter.split(x, y, groups), start=1):
            model = new_model(args)
            model.fit(x[train_indices], y[train_indices])
            scores[valid_indices] = model.predict_proba(x[valid_indices])[:, 1]
            print(f"completed fold {fold}/{args.folds}")
        result = rankings_from_scores(row_keys, scores, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
