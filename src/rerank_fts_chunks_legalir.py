#!/usr/bin/env python3
"""BGE-rerank the actual legal-text chunks returned by SQLite FTS5."""

import argparse
import json
import sqlite3
from pathlib import Path

from fts_chunk_legalir import query_expression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-k", type=int, default=1000)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model-cache", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    query_items = list(queries.items())[args.query_start :]
    if args.max_queries is not None:
        query_items = query_items[: args.max_queries]
    connection = sqlite3.connect(args.database)
    candidates: dict[str, list[tuple[str, str]]] = {}
    pairs = []
    locations = []
    for query_id, row in query_items:
        expression = query_expression(str(row["question"]))
        best_by_document: dict[str, tuple[str, str]] = {}
        if expression:
            matches = connection.execute(
                "SELECT source_id, name, passage FROM chunks WHERE chunks MATCH ? "
                "ORDER BY bm25(chunks, 0.0, 2.0, 1.0) LIMIT ?",
                (expression, args.chunk_k),
            )
            for source_id, name, passage in matches:
                source_id = str(source_id)
                if source_id not in best_by_document:
                    best_by_document[source_id] = (str(name), str(passage))
                if len(best_by_document) == args.candidate_k:
                    break
        rows = [(document_id, f"{name}\n{passage}") for document_id, (name, passage) in best_by_document.items()]
        candidates[str(query_id)] = rows
        for rank, (_, text) in enumerate(rows):
            pairs.append([str(row["question"]), text])
            locations.append((str(query_id), rank))
    connection.close()

    print(f"scoring {len(pairs)} question-to-matched-chunk pairs")
    model = CrossEncoder(args.model, cache_folder=str(args.model_cache))
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    grouped: dict[str, list[tuple[float, int, str]]] = {
        str(query_id): [] for query_id, _ in query_items
    }
    for score, (query_id, rank) in zip(scores, locations):
        document_id = candidates[query_id][rank][0]
        grouped[query_id].append((float(score), rank, document_id))

    result = {}
    score_rows = {}
    for query_id, ranked in grouped.items():
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        result[query_id] = {"answer": [item[2] for item in ranked[: args.top_k]]}
        score_rows[query_id] = [
            {"id": item[2], "score": item[0], "fts_rank": item[1] + 1}
            for item in ranked
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions to {args.output}")
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        args.scores_output.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote reranker scores to {args.scores_output}")


if __name__ == "__main__":
    main()
