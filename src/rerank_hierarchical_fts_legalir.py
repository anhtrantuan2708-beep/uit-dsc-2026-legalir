#!/usr/bin/env python3
"""Rerank hierarchy-aware FTS children with a cross-encoder.

The existing V12 reranker reads the flat FTS schema.  This companion script
reads the `children` table from retrieve_hierarchical_fts_legalir.py and keeps
the legal hierarchy path attached to each matched child.  It never modifies
V10/V12 outputs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from retrieve_hierarchical_fts_legalir import query_expression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-k", type=int, default=1_000)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
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
    candidates: dict[str, list[tuple[str, str, float]]] = {}
    pairs: list[list[str]] = []
    locations: list[tuple[str, int]] = []
    for query_id, row in query_items:
        expression = query_expression(str(row["question"]))
        best_by_source: dict[str, tuple[str, str, float]] = {}
        if expression:
            matches = connection.execute(
                "SELECT source_id, child_id, path, body, -bm25(children, 0.0, 0.0, 0.0, 0.35, 1.0) AS score "
                "FROM children WHERE children MATCH ? ORDER BY score DESC LIMIT ?",
                (expression, args.chunk_k),
            )
            for source_id, child_id, path, body, fts_score in matches:
                source_id = str(source_id)
                if source_id not in best_by_source:
                    best_by_source[source_id] = (str(child_id), f"{path}\n{body}", float(fts_score))
                if len(best_by_source) == args.candidate_k:
                    break

        rows = [(source_id, child_id, text, fts_score) for source_id, (child_id, text, fts_score) in best_by_source.items()]
        candidates[str(query_id)] = [(source_id, child_id, fts_score) for source_id, child_id, _, fts_score in rows]
        for rank, (_, _, text, _) in enumerate(rows):
            pairs.append([str(row["question"]), text])
            locations.append((str(query_id), rank))
    connection.close()

    print(f"scoring {len(pairs)} question-to-hierarchy-child pairs")
    model = CrossEncoder(args.model, cache_folder=str(args.model_cache))
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True) if pairs else []

    grouped: dict[str, list[tuple[float, int, str, str, float]]] = {str(query_id): [] for query_id, _ in query_items}
    for score, (query_id, rank) in zip(scores, locations):
        source_id, child_id, fts_score = candidates[query_id][rank]
        grouped[query_id].append((float(score), rank, source_id, child_id, fts_score))

    result = {}
    score_rows = {}
    for query_id, ranked in grouped.items():
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        result[query_id] = {"answer": [item[2] for item in ranked[: args.top_k]]}
        score_rows[query_id] = [
            {
                "id": source_id,
                "score": score,
                "fts_rank": fts_rank + 1,
                "child_id": child_id,
                "fts_score": fts_score,
            }
            for score, fts_rank, source_id, child_id, fts_score in ranked
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        args.scores_output.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
