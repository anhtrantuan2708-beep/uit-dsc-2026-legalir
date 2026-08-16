#!/usr/bin/env python3
"""Restrict a reranked list to the first N candidates of its source ranking."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("reranked", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    candidates, reranked = load(args.candidates), load(args.reranked)
    result = {}
    for query_id, row in reranked.items():
        allowed = [str(value) for value in candidates[query_id].get("answer", [])[: args.candidate_k]]
        allowed_set = set(allowed)
        answer = [
            str(value)
            for value in row.get("answer", [])
            if str(value) in allowed_set
        ]
        answer.extend(value for value in allowed if value not in answer)
        result[query_id] = {"answer": answer[: args.top_k]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} restricted rankings to {args.output}")


if __name__ == "__main__":
    main()
