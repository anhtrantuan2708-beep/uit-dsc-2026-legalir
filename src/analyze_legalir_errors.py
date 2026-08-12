#!/usr/bin/env python3
"""Create a compact error-analysis report for LegalIR retrieval experiments."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_titles(path: Path) -> dict[str, str]:
    titles = {}
    for file in path.glob("*.json"):
        row = load(file)
        if isinstance(row, dict) and "id" in row:
            titles[str(row["id"])] = str(row.get("name", ""))
    return titles


def hit(gold: set[str], prediction: dict, query_id: str) -> bool:
    predicted = {str(value) for value in prediction.get(query_id, {}).get("answer", [])}
    return bool(gold & predicted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("bm25", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("hybrid", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    gold, bm25, dense, hybrid = load(args.gold), load(args.bm25), load(args.dense), load(args.hybrid)
    titles = corpus_titles(args.corpus)
    counts = {"bm25_only": 0, "dense_only": 0, "both": 0, "neither": 0, "hybrid_rescued": 0}
    missed = []
    partial = []

    for query_id, row in gold.items():
        gold_ids = {str(value) for value in row["answer"]}
        bm25_hit = hit(gold_ids, bm25, query_id)
        dense_hit = hit(gold_ids, dense, query_id)
        hybrid_ids = {str(value) for value in hybrid.get(query_id, {}).get("answer", [])}
        overlap = gold_ids & hybrid_ids

        if bm25_hit and dense_hit:
            counts["both"] += 1
        elif bm25_hit:
            counts["bm25_only"] += 1
        elif dense_hit:
            counts["dense_only"] += 1
        else:
            counts["neither"] += 1
        if overlap and not (bm25_hit and dense_hit):
            counts["hybrid_rescued"] += 1

        example = {
            "query_id": query_id,
            "question": row["question"],
            "gold_ids": sorted(gold_ids),
            "gold_titles": [titles.get(doc_id, "unknown") for doc_id in sorted(gold_ids)],
            "hybrid_ids": hybrid.get(query_id, {}).get("answer", []),
            "hybrid_titles": [titles.get(str(doc_id), "unknown") for doc_id in hybrid.get(query_id, {}).get("answer", [])],
        }
        if not overlap:
            missed.append(example)
        elif overlap != gold_ids:
            partial.append(example)

    report = {
        "queries": len(gold),
        "retriever_hit_overlap": counts,
        "fully_missed_examples": missed[: args.examples],
        "partially_retrieved_examples": partial[: args.examples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"fully missed: {len(missed)}")
    print(f"partial: {len(partial)}")


if __name__ == "__main__":
    main()
