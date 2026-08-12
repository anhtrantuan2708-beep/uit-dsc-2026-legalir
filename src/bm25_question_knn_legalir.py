#!/usr/bin/env python3
"""Lexical nearest-question retriever for labelled LegalIR train data.

Complements dense question KNN: BM25 is especially useful for legal numbers,
document codes, years, and exact procedural names.
"""

import argparse
import json
from collections import Counter, defaultdict
from math import log
from pathlib import Path
import re
import unicodedata

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
STOPWORDS = {"la", "va", "cua", "cho", "voi", "nhung", "duoc", "theo", "trong", "khi", "nao", "gi", "nhu", "mot", "cac", "co", "khong", "tu", "den"}


def tokens(text: str) -> list[str]:
    folded = "".join(char for char in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(char) != "Mn")
    return [token for token in TOKEN_RE.findall(folded) if len(token) > 1 and token not in STOPWORDS]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude a labelled question with the same ID (for train-time rankings).",
    )
    args = parser.parse_args()
    queries, train = load(args.queries), load(args.labelled_train)
    train_ids = list(train)
    postings, document_lengths, document_frequency = defaultdict(list), [], Counter()
    for index, query_id in enumerate(train_ids):
        counts = Counter(tokens(str(train[query_id]["question"])))
        document_lengths.append(sum(counts.values()))
        for term, count in counts.items():
            postings[term].append((index, count))
            document_frequency[term] += 1
    avg_length = sum(document_lengths) / len(document_lengths)
    result = {}
    for query_id, row in queries.items():
        scores = defaultdict(float)
        for term in tokens(str(row["question"])):
            postings_for_term = postings.get(term, [])
            if not postings_for_term:
                continue
            idf = log(1 + (len(train_ids) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            for index, term_frequency in postings_for_term:
                denominator = term_frequency + 1.2 * (1 - 0.75 + 0.75 * document_lengths[index] / avg_length)
                scores[index] += idf * (term_frequency * 2.2 / denominator)
        nearest = sorted(scores, key=lambda index: (-scores[index], train_ids[index]))
        if args.exclude_self:
            nearest = [index for index in nearest if train_ids[index] != query_id]
        nearest = nearest[: args.neighbors]
        document_scores = defaultdict(float)
        for rank, index in enumerate(nearest, start=1):
            for document_id in train[train_ids[index]].get("answer", []):
                document_scores[str(document_id)] += scores[index] / rank
        ranked = sorted(document_scores, key=lambda doc: (-document_scores[doc], doc))[: args.top_k]
        result[query_id] = {"answer": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} lexical nearest-question predictions to {args.output}")


if __name__ == "__main__":
    main()
