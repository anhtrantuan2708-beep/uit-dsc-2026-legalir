#!/usr/bin/env python3
"""Create a deterministic LegalIR split with no gold-document overlap."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("train_output", type=Path)
    parser.add_argument("dev_output", type=Path)
    parser.add_argument("--dev-ratio", type=float, default=0.143)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    query_ids = list(data)
    union_find = UnionFind(query_ids)
    document_owner: dict[str, str] = {}
    for query_id, row in data.items():
        for document_id in map(str, row.get("answer", [])):
            owner = document_owner.setdefault(document_id, query_id)
            union_find.union(query_id, owner)

    components: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        components[union_find.find(query_id)].append(query_id)
    groups = list(components.values())
    random.Random(args.seed).shuffle(groups)

    target = round(len(data) * args.dev_ratio)
    dev_ids: set[str] = set()
    for group in groups:
        if len(dev_ids) >= target:
            break
        current_distance = abs(target - len(dev_ids))
        next_distance = abs(target - (len(dev_ids) + len(group)))
        if next_distance <= current_distance or not dev_ids:
            dev_ids.update(group)

    dev = {query_id: data[query_id] for query_id in query_ids if query_id in dev_ids}
    train = {query_id: data[query_id] for query_id in query_ids if query_id not in dev_ids}
    train_documents = {str(value) for row in train.values() for value in row.get("answer", [])}
    dev_documents = {str(value) for row in dev.values() for value in row.get("answer", [])}
    overlap = train_documents & dev_documents
    if overlap:
        raise RuntimeError(f"document leakage remains: {len(overlap)} IDs")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.dev_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    args.dev_output.write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"train={len(train)} dev={len(dev)} components={len(groups)} "
        f"train_docs={len(train_documents)} dev_docs={len(dev_documents)} overlap=0"
    )


if __name__ == "__main__":
    main()
