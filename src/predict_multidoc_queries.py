#!/usr/bin/env python3
"""Predict whether a LegalIR query is likely to have multiple gold documents."""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()

    train = load(args.labelled_train)
    queries = load(args.queries)
    train_texts = [row.get("question", "") for row in train.values()]
    labels = [int(len(row.get("answer", [])) > 1) for row in train.values()]
    query_ids = list(queries)
    query_texts = [queries[query_id].get("question", "") for query_id in query_ids]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    query_matrix = vectorizer.transform(query_texts)
    model = LogisticRegression(
        class_weight="balanced",
        C=2.0,
        max_iter=500,
        random_state=42,
    )
    model.fit(train_matrix, labels)
    probabilities = model.predict_proba(query_matrix)[:, 1]
    result = {query_id: float(probability) for query_id, probability in zip(query_ids, probabilities)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(result)} probabilities to {args.output}; "
        f"train multi-document rate={sum(labels) / len(labels):.4f}"
    )


if __name__ == "__main__":
    main()
