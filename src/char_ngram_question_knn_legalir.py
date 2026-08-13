#!/usr/bin/env python3
"""Retrieve LegalIR documents from character-ngram-similar train questions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--analyzer", choices=["char_wb", "word"], default="char_wb")
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    queries, train = load(args.queries), load(args.labelled_train)
    train_ids = list(train)
    query_ids = list(queries)
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(args.min_n, args.max_n),
        min_df=args.min_df,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(
        [normalize(str(train[query_id]["question"])) for query_id in train_ids]
    )
    query_matrix = vectorizer.transform(
        [normalize(str(queries[query_id]["question"])) for query_id in query_ids]
    )

    result = {}
    neighbor_count = min(args.neighbors, len(train_ids))
    for index, query_id in enumerate(query_ids):
        similarities = (query_matrix[index] @ train_matrix.T).toarray()[0]
        nearest = np.argpartition(similarities, -neighbor_count)[-neighbor_count:]
        nearest = nearest[np.argsort(similarities[nearest])[::-1]]
        document_scores = defaultdict(float)
        for rank, train_index in enumerate(nearest, start=1):
            similarity = float(similarities[train_index])
            if similarity <= 0:
                continue
            for document_id in train[train_ids[int(train_index)]].get("answer", []):
                document_scores[str(document_id)] += similarity / rank
        ranked = sorted(document_scores, key=lambda doc: (-document_scores[doc], doc))[: args.top_k]
        result[query_id] = {"answer": ranked}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} character-ngram question predictions to {args.output}")


if __name__ == "__main__":
    main()
