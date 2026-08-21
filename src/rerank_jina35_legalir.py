#!/usr/bin/env python3
"""Rerank LegalIR candidates with the local Jina v3.5 MLX listwise model.

The candidate and evidence contract intentionally matches the rejected Qwen3
smoke so the comparison is fair: V12 top-5 plus flat/hierarchy FTS top-20,
with at most one matched chunk from each FTS index per source document.  The
model ranks evidence passages jointly and document score is the maximum score
of its evidence passages.  Completed query rows are cached after every query.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from rerank_qwen3_legalir import (
    atomic_dump,
    candidate_evidence,
    load,
    max_rss_mb,
    ordered_union,
    rank_score_rows,
    swap_usage,
)


def load_mlx_reranker(model_path: Path, block_size: int, max_length: int, max_doc_length: int):
    model_path = model_path.resolve()
    if not (model_path / "rerank.py").exists():
        raise SystemExit(f"missing Jina MLX model code: {model_path / 'rerank.py'}")
    sys.path.insert(0, str(model_path))
    try:
        from rerank import MLXReranker
    finally:
        sys.path.pop(0)
    return MLXReranker(
        model_path=str(model_path),
        projector_path=str(model_path / "projector.safetensors"),
        block_size=block_size,
        max_length=max_length,
        max_query_length=512,
        max_doc_length=max_doc_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("base", type=Path, help="V12 top-5 ranking")
    parser.add_argument("flat_candidates", type=Path, help="flat FTS top-100 ranking")
    parser.add_argument("hierarchy_candidates", type=Path, help="hierarchy FTS top-100 ranking")
    parser.add_argument("flat_database", type=Path)
    parser.add_argument("hierarchy_database", type=Path)
    parser.add_argument("output", type=Path, help="Jina direct top-k ranking")
    parser.add_argument("--base-output", type=Path)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/jina-reranker-v3.5-mlx")
    )
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--fts-k", type=int, default=20)
    parser.add_argument("--hierarchy-k", type=int, default=20)
    parser.add_argument("--flat-chunk-k", type=int, default=1_000)
    parser.add_argument("--hierarchy-chunk-k", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument("--max-doc-length", type=int, default=512)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    if args.top_k > 5 or args.block_size < 1 or args.max_doc_length < 64:
        raise SystemExit("top-k <= 5, block-size >= 1 and max-doc-length >= 64 are required")

    queries, base, flat_candidates, hierarchy_candidates = map(
        load, (args.queries, args.base, args.flat_candidates, args.hierarchy_candidates)
    )
    query_items = list(queries.items())[args.query_start :]
    if args.max_queries is not None:
        query_items = query_items[: args.max_queries]
    if not query_items:
        raise SystemExit("no query rows selected")

    candidates: dict[str, dict[str, list[str]]] = {}
    selected_base: dict[str, dict[str, list[str]]] = {}
    for query_id, _ in query_items:
        query_id = str(query_id)
        selected_base[query_id] = {
            "answer": [str(item) for item in base.get(query_id, {}).get("answer", [])[: args.base_k]]
        }
        candidates[query_id] = {
            "answer": ordered_union(
                selected_base[query_id]["answer"],
                [str(item) for item in flat_candidates.get(query_id, {}).get("answer", [])[: args.fts_k]],
                [str(item) for item in hierarchy_candidates.get(query_id, {}).get("answer", [])[: args.hierarchy_k]],
            )
        }
    atomic_dump(args.candidates_output, candidates)
    if args.base_output:
        atomic_dump(args.base_output, selected_base)

    cached_scores: dict[str, list[dict[str, Any]]] = {}
    if args.scores_output.exists():
        cached_scores = load(args.scores_output)
        expected = {str(query_id) for query_id, _ in query_items}
        cached_scores = {query_id: rows for query_id, rows in cached_scores.items() if query_id in expected}
    old_stats = load(args.stats_output) if args.stats_output.exists() else {}
    old_runtime = old_stats.get("per_query", {})
    runtime_rows: dict[str, dict[str, Any]] = {}

    remaining = [str(query_id) for query_id, _ in query_items if str(query_id) not in cached_scores]
    print(f"selected queries={len(query_items)} cached={len(cached_scores)}")
    model = None
    if remaining:
        model = load_mlx_reranker(
            args.model_path, args.block_size, args.max_length, args.max_doc_length
        )

    import sqlite3

    flat_connection = sqlite3.connect(args.flat_database)
    hierarchy_connection = sqlite3.connect(args.hierarchy_database)
    started = time.perf_counter()
    try:
        for index, (query_id, row) in enumerate(query_items, start=1):
            query_id = str(query_id)
            if query_id in cached_scores:
                runtime_rows[query_id] = dict(old_runtime.get(query_id, {}))
                runtime_rows[query_id]["cached"] = True
                continue

            question = str(row["question"])
            candidate_ids = candidates[query_id]["answer"]
            evidence = candidate_evidence(
                flat_connection,
                hierarchy_connection,
                question,
                candidate_ids,
                args.flat_chunk_k,
                args.hierarchy_chunk_k,
            )
            passages: list[str] = []
            locations: list[tuple[str, str]] = []
            for source_id in candidate_ids:
                for item in evidence[source_id]:
                    passages.append(item["text"])
                    locations.append((source_id, item["source"]))

            query_started = time.perf_counter()
            results = model.rerank(question, passages) if passages else []
            best_scores: dict[str, float] = {}
            evidence_sources: dict[str, list[str]] = {}
            for result in results:
                passage_index = int(result["index"])
                source_id, source_name = locations[passage_index]
                score = float(result["relevance_score"])
                evidence_sources.setdefault(source_id, []).append(source_name)
                if score > best_scores.get(source_id, float("-inf")):
                    best_scores[source_id] = score

            cached_scores[query_id] = [
                {
                    "id": source_id,
                    "score": best_scores.get(source_id),
                    "candidate_rank": rank,
                    "evidence": evidence_sources.get(source_id, []),
                }
                for rank, source_id in enumerate(candidate_ids, start=1)
            ]
            runtime_rows[query_id] = {
                "cached": False,
                "candidate_count": len(candidate_ids),
                "matched_candidate_count": sum(bool(evidence[source_id]) for source_id in candidate_ids),
                "evidence_passage_count": len(passages),
                "seconds": round(time.perf_counter() - query_started, 3),
                "max_rss_mb": max_rss_mb(),
            }
            atomic_dump(args.scores_output, cached_scores)
            print(
                f"[{index}/{len(query_items)}] {query_id}: {len(passages)} passages, "
                f"{runtime_rows[query_id]['seconds']:.2f}s"
            )
    finally:
        flat_connection.close()
        hierarchy_connection.close()

    direct: dict[str, dict[str, list[str]]] = {}
    for query_id, _ in query_items:
        query_id = str(query_id)
        ranked = rank_score_rows(cached_scores[query_id])
        cached_scores[query_id] = ranked
        direct[query_id] = {"answer": [str(item["id"]) for item in ranked[: args.top_k]]}
    atomic_dump(args.scores_output, cached_scores)
    atomic_dump(args.output, direct)
    stats = {
        "model": "jinaai/jina-reranker-v3.5-mlx",
        "model_path": str(args.model_path),
        "queries": len(query_items),
        "block_size": args.block_size,
        "max_length": args.max_length,
        "max_doc_length": args.max_doc_length,
        "total_seconds": round(time.perf_counter() - started, 3),
        "max_rss_mb": max(float(old_stats.get("max_rss_mb", 0.0)), max_rss_mb()),
        "swap_usage": swap_usage(),
        "per_query": runtime_rows,
    }
    atomic_dump(args.stats_output, stats)
    print(f"wrote {len(direct)} direct top-{args.top_k} predictions to {args.output}")
    print(f"max RSS: {stats['max_rss_mb']} MB; swap: {stats['swap_usage']}")


if __name__ == "__main__":
    main()
