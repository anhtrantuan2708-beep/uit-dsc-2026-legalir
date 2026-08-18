#!/usr/bin/env python3
"""Safely promote a candidate only when official-corpus metadata strongly matches.

This is deliberately conservative: a document-number match is trustworthy,
while weak keyword similarity never replaces the retrieval ranking.
"""

import argparse
import json
import re
from pathlib import Path

from legalir_baseline import STOPWORDS

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
NUMBER_RE = re.compile(r"\b\d{1,4}\s*/\s*(?:\d{2,4}\s*/\s*)?[A-Za-zÀ-ỹ0-9._-]{2,40}", re.IGNORECASE)
KINDS = ("luật", "nghị định", "thông tư", "quyết định", "chỉ thị", "nghị quyết", "pháp lệnh")


def normalized_numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", item).upper() for item in NUMBER_RE.findall(text)}


def tokens(text: str) -> set[str]:
    return {item.lower() for item in TOKEN_RE.findall(text) if len(item) >= 4 and item.lower() not in STOPWORDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result, promoted = {}, 0
    for query_id, row in queries.items():
        question = str(row["question"])
        base = [str(item) for item in candidates.get(str(query_id), {}).get("answer", [])[: args.candidate_k]]
        q_numbers = normalized_numbers(question)
        q_kind = next((kind for kind in KINDS if kind in question.lower()), "")
        scored = []
        for rank, document_id in enumerate(base):
            item = metadata.get(document_id, {})
            exact_number = bool(q_numbers & normalized_numbers(str(item.get("document_number", ""))))
            same_kind = bool(q_kind and q_kind == str(item.get("document_type", "")).lower())
            # Title is only a tiebreaker after a document number or legal type match.
            overlap = len(tokens(question) & tokens(str(item.get("title", ""))))
            confidence = 100 if exact_number else (2 + min(overlap, 3) if same_kind and overlap >= 2 else 0)
            scored.append((confidence, rank, document_id))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        if ranked and ranked[0][0] > 0 and ranked[0][1] != 0:
            promoted += 1
        result[str(query_id)] = {"answer": [item[2] for item in ranked[: args.top_k]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions; metadata confidently promoted a candidate for {promoted} queries")


if __name__ == "__main__":
    main()
