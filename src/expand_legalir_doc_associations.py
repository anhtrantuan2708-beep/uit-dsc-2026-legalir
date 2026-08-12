#!/usr/bin/env python3
"""Suggest documents that co-occur with strong candidates in labelled train."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    candidates, train = load(args.candidates), load(args.labelled_train)

    links: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train.values():
        answer = [str(value) for value in row.get("answer", [])]
        for left in answer:
            for right in answer:
                if left != right:
                    links[left][right] += 1

    result = {}
    for query_id, row in candidates.items():
        scores = defaultdict(float)
        for rank, source_id in enumerate(row.get("answer", [])[: args.seed_k], start=1):
            for target_id, count in links[str(source_id)].items():
                scores[target_id] += count / rank
        ranked = sorted(scores, key=lambda doc: (-scores[doc], doc))[: args.top_k]
        result[query_id] = {"answer": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} document-association predictions to {args.output}")


if __name__ == "__main__":
    main()
