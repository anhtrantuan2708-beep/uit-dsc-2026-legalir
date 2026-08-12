#!/usr/bin/env python3
"""Train and apply a lightweight LegalIR top-5 learning-to-rank model.

The model learns from rank positions, retriever agreement, lexical overlap,
document type, and train-document frequency. It never needs hidden Public gold.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DOC_TYPES = ("bo luat", "luat", "nghi dinh", "thong tu", "quyet dinh", "nghi quyet", "chi thi", "phap lenh")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(fold(text)) if len(token) > 1}


def load_corpus(path: Path) -> dict[str, dict]:
    rows = {}
    for file in path.glob("*.json"):
        row = load(file)
        if isinstance(row, dict) and "id" in row:
            rows[str(row["id"])] = row
    return rows


def build_priors(labelled: dict) -> Counter:
    return Counter(str(document_id) for row in labelled.values() for document_id in row.get("answer", []))


def prepare_corpus(corpus: dict[str, dict]) -> dict[str, tuple[set[str], set[str], set[str], list[float]]]:
    prepared = {}
    for document_id, document in corpus.items():
        title = fold(str(document.get("name", "")).replace("-", " "))
        passage = fold(str(document.get("passage", "")))
        prepared[document_id] = (
            tokens(title),
            tokens(passage),
            set(YEAR_RE.findall(title + " " + passage[:1000])),
            [float(kind in title) for kind in DOC_TYPES],
        )
    return prepared


def feature_row(q_tokens: set[str], q_years: set[str], document_id: str, rankings: list[dict], corpus_features: dict, priors: Counter) -> list[float]:
    rank_values = []
    for ranking in rankings:
        values = [str(value) for value in ranking]
        try:
            rank_values.append(values.index(document_id) + 1)
        except ValueError:
            rank_values.append(1000)
    reciprocal = [0.0 if rank == 1000 else 1.0 / rank for rank in rank_values]
    title_tokens, passage_tokens, doc_years, type_flags = corpus_features.get(document_id, (set(), set(), set(), [0.0] * len(DOC_TYPES)))
    title_overlap = len(q_tokens & title_tokens)
    passage_overlap = len(q_tokens & passage_tokens)
    return [
        *reciprocal,
        *[float(rank if rank != 1000 else 200) / 200.0 for rank in rank_values],
        float(sum(rank != 1000 for rank in rank_values)),
        max(reciprocal, default=0.0),
        sum(reciprocal),
        float(title_overlap),
        title_overlap / max(len(q_tokens), 1),
        float(passage_overlap),
        passage_overlap / max(len(q_tokens), 1),
        float(len(q_years & doc_years)),
        float(priors[document_id]),
        *type_flags,
    ]


def candidates_for_query(query_id: str, ranking_files: list[dict], limit: int) -> list[str]:
    score = Counter()
    for ranking in ranking_files:
        for rank, document_id in enumerate(ranking.get(query_id, {}).get("answer", [])[:limit], start=1):
            score[str(document_id)] += 1 / (15 + rank)
    return [document_id for document_id, _ in score.most_common(limit)]


def train(args) -> None:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    labelled = load(args.labelled)
    queries = load(args.queries)
    rankings = [load(path) for path in args.rankings]
    corpus = prepare_corpus(load_corpus(args.corpus))
    priors = build_priors(labelled)
    features, labels = [], []
    positives = negatives = 0
    for query_id, row in queries.items():
        gold = {str(value) for value in row.get("answer", [])}
        candidates = candidates_for_query(query_id, rankings, args.candidate_k)
        ranked_lists = [ranking.get(query_id, {}).get("answer", []) for ranking in rankings]
        q_tokens = tokens(row["question"])
        q_years = set(YEAR_RE.findall(row["question"]))
        positive_docs = [document_id for document_id in candidates if document_id in gold]
        negative_docs = [document_id for document_id in candidates if document_id not in gold][: args.negatives]
        for document_id in positive_docs + negative_docs:
            features.append(feature_row(q_tokens, q_years, document_id, ranked_lists, corpus, priors))
            label = int(document_id in gold)
            labels.append(label)
            positives += label
            negatives += 1 - label
    print(f"training rows: {len(labels)} | positives: {positives} | negatives: {negatives}")
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(np.asarray(features, dtype="float32"), np.asarray(labels))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "priors": priors, "ranking_count": len(rankings)}, args.model)
    print(f"saved ranker to {args.model}")


def predict(args) -> None:
    import joblib
    import numpy as np

    bundle = joblib.load(args.model)
    model, priors = bundle["model"], bundle["priors"]
    queries = load(args.queries)
    rankings = [load(path) for path in args.rankings]
    if len(rankings) != bundle["ranking_count"]:
        raise SystemExit("Ranking count differs from training")
    corpus = prepare_corpus(load_corpus(args.corpus))
    result = {}
    for query_id, row in queries.items():
        candidates = candidates_for_query(query_id, rankings, args.candidate_k)
        ranked_lists = [ranking.get(query_id, {}).get("answer", []) for ranking in rankings]
        q_tokens = tokens(row["question"])
        q_years = set(YEAR_RE.findall(row["question"]))
        values = np.asarray(
            [feature_row(q_tokens, q_years, document_id, ranked_lists, corpus, priors) for document_id in candidates],
            dtype="float32",
        )
        probabilities = model.predict_proba(values)[:, 1]
        ordered = sorted(range(len(candidates)), key=lambda index: (-probabilities[index], index))
        result[query_id] = {"answer": [candidates[index] for index in ordered[: args.top_k]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} learned rankings to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict"):
        command = sub.add_parser(name)
        command.add_argument("queries", type=Path)
        command.add_argument("corpus", type=Path)
        command.add_argument("model", type=Path)
        command.add_argument("rankings", type=Path, nargs="+")
        command.add_argument("--candidate-k", type=int, default=100)
    training = sub.choices["train"]
    training.add_argument("--labelled", type=Path, required=True)
    training.add_argument("--negatives", type=int, default=30)
    prediction = sub.choices["predict"]
    prediction.add_argument("output", type=Path)
    prediction.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
