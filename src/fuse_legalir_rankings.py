#!/usr/bin/env python3
"""Fuse two LegalIR ranking files using Reciprocal Rank Fusion (RRF)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path, help="First ranking JSON, e.g. BM25")
    parser.add_argument("second", type=Path, help="Second ranking JSON, e.g. dense")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--first-weight", type=float, default=1.0, help="BM25 vote weight")
    parser.add_argument("--second-weight", type=float, default=1.0, help="Dense vote weight")
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    result = {}
    for query_id in first:
        scores = defaultdict(float)
        for ranking, weight in [(first, args.first_weight), (second, args.second_weight)]:
            for rank, document_id in enumerate(ranking.get(query_id, {}).get("answer", []), start=1):
                scores[str(document_id)] += weight / (args.rrf_k + rank)
        # Longer lists are allowed for internal reranking. Submission validation
        # enforces the final five-ID limit.
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[: args.top_k]
        result[query_id] = {"answer": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} fused predictions to {args.output}")


if __name__ == "__main__":
    main()
