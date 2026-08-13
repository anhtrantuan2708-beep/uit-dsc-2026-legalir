#!/usr/bin/env python3
"""Retrieve legal documents via nearest labelled training questions.

This is a supervised retriever: if a new question resembles a labelled train
question, its associated legal-document IDs are strong candidates.  Dev and
Public questions are never used to build this index.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path, help="dev or Public questions")
    parser.add_argument("labelled_train", type=Path, help="train questions with answer IDs")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--model-cache", type=Path, default=Path("models/e5-small"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/derived/e5_train_question_embeddings.npy"))
    parser.add_argument("--embedding-ids", type=Path, default=Path("data/derived/e5_train_question_ids.json"))
    parser.add_argument("--query-embeddings", type=Path)
    parser.add_argument("--query-embedding-ids", type=Path)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--aggregation",
        choices=["rank_vote", "max_similarity"],
        default="rank_vote",
        help="How labelled neighbours vote for a document.",
    )
    parser.add_argument("--similarity-power", type=float, default=1.0)
    parser.add_argument("--rank-power", type=float, default=1.0)
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude a labelled question with the same ID (for train-time rankings).",
    )
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    queries, train = load(args.queries), load(args.labelled_train)
    train_ids = list(train)
    model = None

    if args.embeddings.exists() and args.embedding_ids.exists() and load(args.embedding_ids) == train_ids:
        print("reusing local labelled-question embeddings")
        train_embeddings = np.load(args.embeddings)
    else:
        model = SentenceTransformer(args.model, cache_folder=str(args.model_cache))
        print(f"encoding {len(train_ids)} labelled training questions locally")
        train_embeddings = model.encode(
            [f"query: {train[query_id]['question']}" for query_id in train_ids],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings, train_embeddings)
        args.embedding_ids.write_text(json.dumps(train_ids), encoding="utf-8")

    query_ids = list(queries)
    if (
        args.query_embeddings
        and args.query_embedding_ids
        and args.query_embeddings.exists()
        and args.query_embedding_ids.exists()
        and load(args.query_embedding_ids) == query_ids
    ):
        print("reusing local query embeddings")
        query_embeddings = np.load(args.query_embeddings)
    else:
        if model is None:
            model = SentenceTransformer(args.model, cache_folder=str(args.model_cache))
        query_embeddings = model.encode(
            [f"query: {queries[query_id]['question']}" for query_id in query_ids],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if args.query_embeddings and args.query_embedding_ids:
            args.query_embeddings.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.query_embeddings, query_embeddings)
            args.query_embedding_ids.write_text(json.dumps(query_ids), encoding="utf-8")
    scores = query_embeddings @ train_embeddings.T
    result = {}
    neighbor_count = min(args.neighbors + (1 if args.exclude_self else 0), len(train_ids))
    for index, query_id in enumerate(query_ids):
        indices = np.argpartition(scores[index], -neighbor_count)[-neighbor_count:]
        indices = indices[np.argsort(scores[index][indices])[::-1]]
        if args.exclude_self:
            indices = [item for item in indices if train_ids[int(item)] != query_id][: args.neighbors]
        document_scores = defaultdict(float)
        for rank, train_index in enumerate(indices, start=1):
            # Similarity carries semantic confidence; rank provides stability.
            similarity = max(float(scores[index][train_index]), 0.0)
            vote = similarity**args.similarity_power / rank**args.rank_power
            for document_id in train[train_ids[int(train_index)]].get("answer", []):
                document_id = str(document_id)
                if args.aggregation == "max_similarity":
                    document_scores[document_id] = max(document_scores[document_id], float(scores[index][train_index]))
                else:
                    document_scores[document_id] += vote
        ranked = sorted(document_scores, key=lambda doc: (-document_scores[doc], doc))[: args.top_k]
        result[query_id] = {"answer": ranked}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} nearest-question predictions to {args.output}")


if __name__ == "__main__":
    main()
