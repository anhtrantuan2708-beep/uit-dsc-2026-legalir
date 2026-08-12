#!/usr/bin/env python3
"""Fuse any number of internal LegalIR rankings with weighted RRF."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("rankings", type=Path, nargs="+", help="ranking JSON files")
    parser.add_argument("--weights", default="", help="comma-separated weight per input, e.g. 1,1,1")
    parser.add_argument("--rrf-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    rows = [load(path) for path in args.rankings]
    weights = [float(value) for value in args.weights.split(",") if value] or [1.0] * len(rows)
    if len(weights) != len(rows):
        raise SystemExit("weights must match the number of ranking files")

    result = {}
    for query_id in rows[0]:
        scores = defaultdict(float)
        for ranking, weight in zip(rows, weights):
            for rank, document_id in enumerate(ranking.get(query_id, {}).get("answer", []), start=1):
                scores[str(document_id)] += weight / (args.rrf_k + rank)
        ranked = sorted(scores, key=lambda doc: (-scores[doc], doc))[: args.top_k]
        result[query_id] = {"answer": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} fused predictions to {args.output}")


if __name__ == "__main__":
    main()
