#!/usr/bin/env python3
"""Add a uniquely resolved explicit document-number candidate to a FTS union.

This is intentionally a narrow retrieval-only rescue, not a reranker: it never
uses labels to choose a document and does nothing unless a number written in a
question resolves to exactly one official-corpus metadata record.  When it
does rescue, the new ID is placed first and the trusted input candidate list is
otherwise kept in its original order.  The candidate budget remains fixed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Includes common Vietnamese legal identifiers such as 115/2015/ND-CP and
# 51/2014/QD-TTg.  Diacritics are removed by normalize_number below.
NUMBER_RE = re.compile(
    r"(?<![A-Z0-9])\d{1,4}\s*/\s*\d{2,4}(?:\s*/\s*[A-ZÀ-Ỹ0-9._-]{2,40})+",
    re.IGNORECASE,
)


def normalize_number(value: str) -> str:
    value = value.upper().replace("Đ", "D")
    value = re.sub(r"\s+", "", value)
    return value.replace("–", "-")


def numbers_in(text: str) -> set[str]:
    return {normalize_number(value) for value in NUMBER_RE.findall(text)}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_coverage(gold: dict, predictions: dict) -> tuple[float, float]:
    recalls: list[float] = []
    hits = 0
    for query_id, row in gold.items():
        wanted = {str(value) for value in row.get("answer", [])}
        returned = set(predictions.get(str(query_id), {}).get("answer", []))
        overlap = len(wanted & returned)
        recalls.append(overlap / len(wanted) if wanted else 1.0)
        hits += int(bool(overlap))
    count = len(recalls) or 1
    return sum(recalls) / count, hits / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("candidates", type=Path, help="Existing flat+hie FTS candidate union")
    parser.add_argument("metadata", type=Path, help="Official-corpus metadata JSON")
    parser.add_argument("output", type=Path)
    parser.add_argument("--gold", type=Path, help="Optional labelled smoke split for coverage only")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")

    queries, candidates, metadata = map(load, (args.queries, args.candidates, args.metadata))
    number_to_ids: dict[str, set[str]] = {}
    for source_id, row in metadata.items():
        for number in numbers_in(str(row.get("document_number", ""))):
            number_to_ids.setdefault(number, set()).add(str(source_id))

    result: dict[str, dict[str, list[str]]] = {}
    explicit_queries = uniquely_resolved = inserted = 0
    for query_id, row in queries.items():
        base = [str(value) for value in candidates.get(str(query_id), {}).get("answer", [])]
        exact_ids: set[str] = set()
        query_numbers = numbers_in(str(row.get("question", "")))
        explicit_queries += int(bool(query_numbers))
        for number in query_numbers:
            matched = number_to_ids.get(number, set())
            if len(matched) == 1:
                exact_ids.update(matched)
        uniquely_resolved += int(bool(exact_ids))
        # At most one fresh ID: this avoids changing the candidate distribution
        # for multi-reference questions and keeps the fixed top-k budget honest.
        fresh = sorted(exact_ids - set(base))[:1]
        inserted += len(fresh)
        merged = fresh + [source_id for source_id in base if source_id not in fresh]
        result[str(query_id)] = {"answer": merged[: args.top_k]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} rescued rankings to {args.output}")
    print(f"queries_with_number: {explicit_queries}")
    print(f"queries_uniquely_resolved: {uniquely_resolved}")
    print(f"fresh_candidates_inserted: {inserted}")
    if args.gold:
        baseline = {
            str(query_id): {"answer": [str(value) for value in row.get("answer", [])[: args.top_k]]}
            for query_id, row in candidates.items()
        }
        gold = load(args.gold)
        before_recall, before_hit = score_coverage(gold, baseline)
        after_recall, after_hit = score_coverage(gold, result)
        print(f"candidate_macro_recall@{args.top_k}: {before_recall:.4f} -> {after_recall:.4f}")
        print(f"candidate_hit@{args.top_k}: {before_hit:.4f} -> {after_hit:.4f}")


if __name__ == "__main__":
    main()
