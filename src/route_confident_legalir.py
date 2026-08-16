#!/usr/bin/env python3
"""Use an alternate LegalIR ranking only when its confidence gate passes."""

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--min-margin", type=float, default=0.002)
    parser.add_argument("--keep-base", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    base, alternate, score_rows = load(args.base), load(args.alternate), load(args.scores)
    result = {}
    routed = 0
    for query_id, row in base.items():
        base_answer = [str(value) for value in row.get("answer", [])]
        ranked_scores = score_rows.get(query_id, [])
        confidence = float(ranked_scores[0]["score"]) if ranked_scores else float("-inf")
        margin = (
            confidence - float(ranked_scores[1]["score"])
            if len(ranked_scores) > 1
            else float("inf")
        )
        should_route = confidence >= args.min_score and margin >= args.min_margin
        sources = [base_answer]
        if should_route:
            sources = [
                base_answer[: args.keep_base],
                alternate.get(query_id, {}).get("answer", []),
                base_answer,
            ]
            routed += 1
        answer = []
        for source in sources:
            for document_id in source:
                document_id = str(document_id)
                if document_id not in answer:
                    answer.append(document_id)
                if len(answer) == args.top_k:
                    break
            if len(answer) == args.top_k:
                break
        result[query_id] = {"answer": answer}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions; routed {routed} queries")


if __name__ == "__main__":
    main()
