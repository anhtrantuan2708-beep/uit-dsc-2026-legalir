#!/usr/bin/env python3
"""Rerank nearby labelled questions with BGE, then transfer document IDs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cached_subset(embedding_path: Path, ids_path: Path, desired_ids: list[str]):
    import numpy as np

    if not embedding_path.exists() or not ids_path.exists():
        return None
    cached_ids = load(ids_path)
    positions = {query_id: index for index, query_id in enumerate(cached_ids)}
    if not all(query_id in positions for query_id in desired_ids):
        return None
    values = np.load(embedding_path)
    return values[[positions[query_id] for query_id in desired_ids]]


def rank_documents(
    matches: list[tuple[float, str]], train: dict, top_k: int
) -> dict[str, list[str]]:
    import numpy as np

    max_scores: dict[str, float] = {}
    rank_votes = defaultdict(float)
    softmax_votes = defaultdict(float)
    logits = np.asarray([score for score, _ in matches], dtype=float)
    weights = np.exp(logits - logits.max())
    weights /= max(weights.sum(), 1e-12)

    for rank, ((score, train_id), weight) in enumerate(zip(matches, weights), start=1):
        for document_id in train[train_id].get("answer", []):
            document_id = str(document_id)
            max_scores[document_id] = max(max_scores.get(document_id, float("-inf")), score)
            rank_votes[document_id] += 1.0 / rank
            softmax_votes[document_id] += float(weight)

    def ordered(scores: dict[str, float]) -> list[str]:
        return sorted(scores, key=lambda doc: (-scores[doc], doc))[:top_k]

    return {
        "max": ordered(max_scores),
        "rank_vote": ordered(rank_votes),
        "softmax": ordered(softmax_votes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--candidate-neighbors", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--e5-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--e5-cache", type=Path, default=Path("models/e5-small"))
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--train-embedding-ids", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-embedding-ids", type=Path, required=True)
    parser.add_argument("--bge-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--bge-cache", type=Path, default=Path("models/bge-reranker-v2-m3"))
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

    queries, train = load(args.queries), load(args.labelled_train)
    query_ids, train_ids = list(queries), list(train)
    train_embeddings = cached_subset(
        args.train_embeddings, args.train_embedding_ids, train_ids
    )
    query_embeddings = cached_subset(
        args.query_embeddings, args.query_embedding_ids, query_ids
    )
    encoder = None
    if train_embeddings is None or query_embeddings is None:
        encoder = SentenceTransformer(args.e5_model, cache_folder=str(args.e5_cache))
    if train_embeddings is None:
        train_embeddings = encoder.encode(
            [f"query: {train[item]['question']}" for item in train_ids],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    if query_embeddings is None:
        query_embeddings = encoder.encode(
            [f"query: {queries[item]['question']}" for item in query_ids],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    neighbor_count = min(args.candidate_neighbors, len(train_ids))
    similarities = query_embeddings @ train_embeddings.T
    neighbor_indices = []
    pairs = []
    for query_index, query_id in enumerate(query_ids):
        indices = np.argpartition(similarities[query_index], -neighbor_count)[-neighbor_count:]
        indices = indices[np.argsort(similarities[query_index][indices])[::-1]]
        neighbor_indices.append(indices)
        pairs.extend(
            [queries[query_id]["question"], train[train_ids[int(index)]]["question"]]
            for index in indices
        )

    print(f"scoring {len(pairs)} query-to-labelled-question pairs")
    reranker = CrossEncoder(args.bge_model, cache_folder=str(args.bge_cache))
    scores = reranker.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)

    outputs = {name: {} for name in ("max", "rank_vote", "softmax")}
    offset = 0
    for query_id, indices in zip(query_ids, neighbor_indices):
        matches = [
            (float(scores[offset + rank]), train_ids[int(index)])
            for rank, index in enumerate(indices)
        ]
        offset += len(indices)
        matches.sort(key=lambda item: (-item[0], item[1]))
        rankings = rank_documents(matches, train, args.top_k)
        for name, answer in rankings.items():
            outputs[name][query_id] = {"answer": answer}

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, result in outputs.items():
        path = args.output_prefix.with_name(f"{args.output_prefix.name}_{name}.json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {len(result)} predictions to {path}")


if __name__ == "__main__":
    main()
