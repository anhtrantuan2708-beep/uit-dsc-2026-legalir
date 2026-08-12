#!/usr/bin/env python3
"""Keep trusted base ranks, then fill remaining slots from a reranker."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("reranked", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-base", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    base, reranked = load(args.base), load(args.reranked)
    result = {}
    for query_id, row in base.items():
        answer = []
        for source in [row.get("answer", [])[: args.keep_base], reranked.get(query_id, {}).get("answer", []), row.get("answer", [])]:
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
    print(f"wrote {len(result)} blended predictions to {args.output}")


if __name__ == "__main__":
    main()
