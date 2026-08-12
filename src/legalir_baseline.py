#!/usr/bin/env python3
"""Small dependency-free BM25 baseline for LegalIR.

Corpus input can be a directory of context_*.json files or a JSON list/object
containing records with id and passage fields. This is a warmup baseline, not
the final competition system.
"""

import argparse
import heapq
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "anh", "bao", "bị", "bởi", "các", "cần", "cho", "có", "của",
    "cùng", "cũng", "đã", "đang", "để", "đến", "đều", "do", "đó", "được", "gì",
    "hay", "không", "khi", "là", "lại", "mà", "một", "nào", "này", "nên", "người",
    "như", "những", "ở", "phải", "sau", "sẽ", "sự", "theo", "thì", "trên", "trong",
    "từ", "tại", "và", "về", "vì", "với", "vào", "việc", "với", "vừa", "yêu",
}


def fold(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d")


FOLDED_STOPWORDS = {fold(word) for word in STOPWORDS}


def tokens(text: str, *, fold_accents: bool, remove_stopwords: bool) -> list[str]:
    value = fold(text) if fold_accents else text.lower()
    values = TOKEN_RE.findall(value)
    stopwords = FOLDED_STOPWORDS if fold_accents else STOPWORDS
    return [value for value in values if not remove_stopwords or value not in stopwords]


def load_queries(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> list[dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records: list[dict] = []
    for file in files:
        value = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(value, list):
            records.extend(value)
        elif isinstance(value, dict) and "passage" in value:
            records.append(value)
        elif isinstance(value, dict):
            records.extend(item for item in value.values() if isinstance(item, dict))
    return [item for item in records if "id" in item and "passage" in item]


def load_query_profiles(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(document_id): values for document_id, values in raw.items()}


class Bm25Index:
    """In-memory BM25 index built once, then reused for every query."""

    def __init__(
        self,
        corpus: list[dict],
        k1: float = 1.2,
        b: float = 0.75,
        include_title: bool = False,
        fold_accents: bool = False,
        remove_stopwords: bool = False,
        query_profiles: dict[str, list[str]] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.include_title = include_title
        self.fold_accents = fold_accents
        self.remove_stopwords = remove_stopwords
        self.query_profiles = query_profiles or {}
        self.doc_ids = [str(item["id"]) for item in corpus]
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}

        for doc_index, item in enumerate(corpus):
            document_text = str(item["passage"])
            if include_title:
                document_text = f"{item.get('name', '')} {document_text}"
            document_text = f"{document_text} {' '.join(self.query_profiles.get(str(item['id']), []))}"
            counts = Counter(
                tokens(
                    document_text,
                    fold_accents=fold_accents,
                    remove_stopwords=remove_stopwords,
                )
            )
            self.doc_lengths.append(sum(counts.values()))
            for term, term_frequency in counts.items():
                self.postings.setdefault(term, []).append((doc_index, term_frequency))

        self.size = len(self.doc_ids)
        self.average_length = sum(self.doc_lengths) / max(self.size, 1)

    def rank(self, query: str, top_k: int) -> list[str]:
        scores: dict[int, float] = {}
        for term in tokens(
            query,
            fold_accents=self.fold_accents,
            remove_stopwords=self.remove_stopwords,
        ):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            document_frequency = len(postings)
            idf = math.log(1 + (self.size - document_frequency + 0.5) / (document_frequency + 0.5))
            for doc_index, term_frequency in postings:
                length_ratio = self.doc_lengths[doc_index] / max(self.average_length, 1)
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (
                    term_frequency * (self.k1 + 1) / denominator
                )

        ranked = heapq.nsmallest(
            top_k,
            scores,
            key=lambda index: (-scores[index], self.doc_ids[index]),
        )
        return [self.doc_ids[index] for index in ranked[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--fold-accents", action="store_true")
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument(
        "--query-profiles",
        type=Path,
        help="JSON document_id -> labelled training questions for document expansion",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    corpus = load_corpus(args.corpus)
    if not corpus:
        raise SystemExit("No corpus records with id + passage were found")
    index = Bm25Index(
        corpus,
        include_title=args.include_title,
        fold_accents=args.fold_accents,
        remove_stopwords=args.remove_stopwords,
        query_profiles=load_query_profiles(args.query_profiles),
    )
    print(f"indexed {index.size} corpus passages")
    result = {
        # Internal candidate files may keep more than 5 ranks for RRF.  Only the
        # final submission is constrained to five IDs by the validator.
        query_id: {"answer": index.rank(row["question"], args.top_k)}
        for query_id, row in queries.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} query predictions to {args.output}")


if __name__ == "__main__":
    main()
