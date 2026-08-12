#!/usr/bin/env python3
"""Measure how much Recall is lost during candidate generation vs final ranking."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("selected", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--examples", type=int, default=30)
    args = parser.parse_args()

    gold = load(args.gold)
    selected = load(args.selected)
    candidates = load(args.candidates)
    cutoffs = sorted(set(args.cutoffs))
    recalls = {cutoff: [] for cutoff in cutoffs}
    selected_recalls = []
    recoverable_queries = 0
    unrecoverable_queries = 0
    missed_rank_bands = Counter()
    examples = []

    for query_id, row in gold.items():
        gold_ids = [str(value) for value in row.get("answer", [])]
        gold_set = set(gold_ids)
        selected_ids = [str(value) for value in selected.get(query_id, {}).get("answer", [])[:5]]
        candidate_ids = [str(value) for value in candidates.get(query_id, {}).get("answer", [])]
        rank = {document_id: index + 1 for index, document_id in enumerate(candidate_ids)}

        selected_hit = gold_set.intersection(selected_ids)
        selected_recalls.append(len(selected_hit) / len(gold_set) if gold_set else 0.0)
        for cutoff in cutoffs:
            recalls[cutoff].append(
                len(gold_set.intersection(candidate_ids[:cutoff])) / len(gold_set) if gold_set else 0.0
            )

        missed = [document_id for document_id in gold_ids if document_id not in selected_hit]
        if not missed:
            continue
        missed_ranks = {document_id: rank.get(document_id) for document_id in missed}
        if any(value is not None for value in missed_ranks.values()):
            recoverable_queries += 1
        else:
            unrecoverable_queries += 1
        for value in missed_ranks.values():
            if value is None:
                missed_rank_bands["outside_top100"] += 1
            elif value <= 10:
                missed_rank_bands["rank_1_10"] += 1
            elif value <= 20:
                missed_rank_bands["rank_11_20"] += 1
            elif value <= 50:
                missed_rank_bands["rank_21_50"] += 1
            else:
                missed_rank_bands["rank_51_100"] += 1
        examples.append(
            {
                "query_id": query_id,
                "question": row.get("question", ""),
                "gold_ids": gold_ids,
                "selected_ids": selected_ids,
                "missed_gold_candidate_ranks": missed_ranks,
            }
        )

    examples.sort(
        key=lambda item: min(
            (rank for rank in item["missed_gold_candidate_ranks"].values() if rank is not None),
            default=10**9,
        )
    )
    report = {
        "queries": len(gold),
        "selected_recall_at_5": round(mean(selected_recalls), 6),
        "candidate_recall": {
            f"at_{cutoff}": round(mean(recalls[cutoff]), 6) for cutoff in cutoffs
        },
        "queries_with_missed_gold_recoverable_in_top100": recoverable_queries,
        "queries_with_all_missed_gold_outside_top100": unrecoverable_queries,
        "missed_gold_rank_bands": dict(missed_rank_bands),
        "nearest_recoverable_examples": examples[: args.examples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "nearest_recoverable_examples"}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
