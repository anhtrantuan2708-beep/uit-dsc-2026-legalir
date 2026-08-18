#!/usr/bin/env python3
"""Evaluate confidence-router settings for a chunk-aware LegalIR ranking.

This is a diagnostic tool: it evaluates only IDs that exist in all inputs and
prints the best configuration.  It never writes a submission or touches the
baseline predictions.
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def routed_answer(base: list[str], alternate: list[str], should_route: bool, keep_base: int, top_k: int) -> list[str]:
    sources = [base] if not should_route else [base[:keep_base], alternate, base]
    answer: list[str] = []
    for source in sources:
        for document_id in source:
            document_id = str(document_id)
            if document_id not in answer:
                answer.append(document_id)
            if len(answer) == top_k:
                return answer
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99])
    parser.add_argument("--keep-base", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    gold, base, alternate, scores = (load(path) for path in (args.gold, args.base, args.alternate, args.scores))
    query_ids = [query_id for query_id in gold if query_id in base and query_id in alternate and query_id in scores]
    if not query_ids:
        raise SystemExit("No query IDs are shared by gold, base, alternate and scores")

    rows = []
    for threshold in args.thresholds:
        for keep_base in args.keep_base:
            recalls, precisions, routed = [], [], 0
            for query_id in query_ids:
                base_answer = [str(value) for value in base[query_id].get("answer", [])]
                alternate_answer = [str(value) for value in alternate[query_id].get("answer", [])]
                row_scores = scores[query_id]
                should_route = bool(row_scores) and float(row_scores[0]["score"]) >= threshold
                routed += int(should_route)
                prediction = set(routed_answer(base_answer, alternate_answer, should_route, keep_base, args.top_k))
                answer = {str(value) for value in gold[query_id].get("answer", [])}
                hits = len(prediction & answer)
                recalls.append(hits / len(answer) if answer else 0.0)
                precisions.append(hits / len(prediction) if prediction else 0.0)
            rows.append((sum(recalls) / len(query_ids), sum(precisions) / len(query_ids), threshold, keep_base, routed))

    print(f"queries evaluated: {len(query_ids)}")
    for recall, precision, threshold, keep_base, routed in sorted(rows, reverse=True):
        print(
            f"recall={recall:.4f} precision={precision:.4f} "
            f"threshold={threshold:.3f} keep_base={keep_base} routed={routed}"
        )


if __name__ == "__main__":
    main()
