#!/usr/bin/env python3
"""Measure whether one-hop legal references can recover missing LegalIR gold IDs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def recall(gold: set[str], candidates: set[str]) -> float:
    return len(gold & candidates) / len(gold) if gold else 1.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--seed-k", type=int, default=20)
    parser.add_argument("--hops", type=int, default=1, choices=(1,))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    rankings = json.loads(args.ranking.read_text(encoding="utf-8"))
    graph = json.loads(args.graph.read_text(encoding="utf-8"))

    buckets: dict[str, dict[str, list[float] | int]] = defaultdict(lambda: {"base": [], "expanded": [], "new_gold": 0, "queries": 0})
    relation_hits: dict[str, int] = defaultdict(int)
    examples = []
    for query_id, row in gold_rows.items():
        gold = {str(item) for item in row.get("answer", [])}
        seeds = [str(item) for item in rankings.get(str(query_id), {}).get("answer", [])[: args.seed_k]]
        expanded = set(seeds)
        edges = []
        for seed in seeds:
            for edge in graph.get(seed, []):
                target = str(edge["target_id"])
                expanded.add(target)
                edges.append(edge)
        base_recall = recall(gold, set(seeds))
        expanded_recall = recall(gold, expanded)
        bucket_names = ["overall", "multi_gold" if len(gold) > 1 else "single_gold"]
        for bucket in bucket_names:
            buckets[bucket]["base"].append(base_recall)
            buckets[bucket]["expanded"].append(expanded_recall)
            buckets[bucket]["new_gold"] += len((gold & expanded) - set(seeds))
            buckets[bucket]["queries"] += 1
        for edge in edges:
            if str(edge["target_id"]) in gold and str(edge["target_id"]) not in seeds:
                relation_hits[str(edge["relation"])] += 1
        if expanded_recall > base_recall and len(examples) < 30:
            examples.append({
                "query_id": str(query_id),
                "gold": sorted(gold),
                "seed_ids": seeds,
                "new_gold_ids": sorted((gold & expanded) - set(seeds)),
            })

    summary = {}
    for name, bucket in buckets.items():
        base = mean(bucket["base"])
        expanded = mean(bucket["expanded"])
        summary[name] = {
            "queries": bucket["queries"],
            "base_recall": base,
            "one_hop_recall": expanded,
            "delta": expanded - base,
            "new_gold_ids": bucket["new_gold"],
        }
    result = {
        "seed_k": args.seed_k,
        "hops": args.hops,
        "summary": summary,
        "relation_hits": dict(sorted(relation_hits.items())),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
