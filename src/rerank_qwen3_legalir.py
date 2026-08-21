#!/usr/bin/env python3
"""Rerank LegalIR candidates with Qwen3 using matched FTS evidence only.

This runner deliberately does not retrieve or embed the corpus.  For each
query it combines a trusted V12 top-5 with the first 20 flat-FTS and hierarchy
FTS candidates, scores at most two matched chunks per document (one from each
index), and caches completed query scores after every query.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fts_chunk_legalir import query_expression


DEFAULT_INSTRUCTION = (
    "Given a Vietnamese legal question, judge whether the document contains "
    "legal provisions necessary to answer it."
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def swap_usage() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def max_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def ordered_union(*rankings: list[str]) -> list[str]:
    result: list[str] = []
    for ranking in rankings:
        for source_id in ranking:
            source_id = str(source_id)
            if source_id not in result:
                result.append(source_id)
    return result


def matched_evidence(
    connection: sqlite3.Connection,
    sql: str,
    expression: str,
    wanted: set[str],
    limit: int,
    source_name: str,
) -> dict[str, list[dict[str, str]]]:
    """Return at most one evidence chunk per document for one index."""
    found: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not expression or not wanted:
        return found
    for source_id, prefix, body in connection.execute(sql, (expression, limit)):
        source_id = str(source_id)
        if source_id not in wanted or found[source_id]:
            continue
        text = f"{prefix}\n{body}".strip()
        if text:
            found[source_id].append({"source": source_name, "text": text})
        if len(found) == len(wanted):
            break
    return found


def candidate_evidence(
    flat_connection: sqlite3.Connection,
    hierarchy_connection: sqlite3.Connection,
    question: str,
    candidate_ids: list[str],
    flat_chunk_k: int,
    hierarchy_chunk_k: int,
) -> dict[str, list[dict[str, str]]]:
    expression = query_expression(question)
    wanted = set(candidate_ids)
    flat = matched_evidence(
        flat_connection,
        "SELECT source_id, name, passage FROM chunks WHERE chunks MATCH ? "
        "ORDER BY bm25(chunks, 0.0, 2.0, 1.0) LIMIT ?",
        expression,
        wanted,
        flat_chunk_k,
        "flat_fts",
    )
    hierarchy = matched_evidence(
        hierarchy_connection,
        "SELECT source_id, path, body FROM children WHERE children MATCH ? "
        "ORDER BY bm25(children, 0.0, 0.0, 0.0, 0.35, 1.0) LIMIT ?",
        expression,
        wanted,
        hierarchy_chunk_k,
        "hierarchy_fts",
    )
    result: dict[str, list[dict[str, str]]] = {}
    for source_id in candidate_ids:
        # Exactly one flat and one hierarchy chunk at most: never rerank a long document.
        result[source_id] = (flat.get(source_id, []) + hierarchy.get(source_id, []))[:2]
    return result


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def rank_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            item["score"] is None,
            -(float(item["score"]) if item["score"] is not None else 0.0),
            int(item["candidate_rank"]),
            str(item["id"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("base", type=Path, help="V12 top-5 ranking")
    parser.add_argument("flat_candidates", type=Path, help="flat FTS top-100 ranking")
    parser.add_argument("hierarchy_candidates", type=Path, help="hierarchy FTS top-100 ranking")
    parser.add_argument("flat_database", type=Path)
    parser.add_argument("hierarchy_database", type=Path)
    parser.add_argument("output", type=Path, help="Qwen direct top-k ranking")
    parser.add_argument("--base-output", type=Path, help="selected V12 rows for fair local evaluation")
    parser.add_argument("--scores-output", type=Path, required=True, help="resumable per-document Qwen scores")
    parser.add_argument("--candidates-output", type=Path, required=True, help="union candidates before reranking")
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--model-cache", type=Path, default=Path("models/qwen3-reranker-0.6b"))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--fts-k", type=int, default=20)
    parser.add_argument("--hierarchy-k", type=int, default=20)
    parser.add_argument("--flat-chunk-k", type=int, default=1_000)
    parser.add_argument("--hierarchy-chunk-k", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 64 or args.top_k > 5:
        raise SystemExit("batch-size >= 1, max-length >= 64, and top-k <= 5 are required")

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
        merged = ordered_union(
            selected_base[query_id]["answer"],
            [str(item) for item in flat_candidates.get(query_id, {}).get("answer", [])[: args.fts_k]],
            [str(item) for item in hierarchy_candidates.get(query_id, {}).get("answer", [])[: args.hierarchy_k]],
        )
        candidates[query_id] = {"answer": merged}
    atomic_dump(args.candidates_output, candidates)
    if args.base_output:
        atomic_dump(args.base_output, selected_base)

    cached_scores: dict[str, list[dict[str, Any]]] = {}
    if args.scores_output.exists():
        cached_scores = load(args.scores_output)
        expected_ids = {str(query_id) for query_id, _ in query_items}
        cached_scores = {query_id: rows for query_id, rows in cached_scores.items() if query_id in expected_ids}
    previous_stats: dict[str, Any] = load(args.stats_output) if args.stats_output.exists() else {}

    device = resolve_device(args.device)
    print(f"selected queries={len(query_items)} device={device} max_length={args.max_length}")
    print(f"reusing cached scores for {len(cached_scores)} queries")

    remaining = [str(query_id) for query_id, _ in query_items if str(query_id) not in cached_scores]
    model = None
    if remaining:
        from sentence_transformers import CrossEncoder
        import torch

        model_kwargs: dict[str, Any] = {}
        if device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        model = CrossEncoder(
            args.model,
            cache_folder=str(args.model_cache),
            device=device,
            max_length=args.max_length,
            prompts={"legal": args.instruction},
            default_prompt_name="legal",
            model_kwargs=model_kwargs,
        )
    else:
        print("all selected query scores are cached; skipping model load")

    flat_connection = sqlite3.connect(args.flat_database)
    hierarchy_connection = sqlite3.connect(args.hierarchy_database)
    previous_runtime = previous_stats.get("per_query", {})
    runtime_rows: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        for index, (query_id, row) in enumerate(query_items, start=1):
            query_id = str(query_id)
            if query_id in cached_scores:
                runtime_rows[query_id] = dict(previous_runtime.get(query_id, {}))
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
            pairs: list[list[str]] = []
            locations: list[tuple[str, int]] = []
            for source_id in candidate_ids:
                for evidence_index, item in enumerate(evidence[source_id]):
                    pairs.append([question, item["text"]])
                    locations.append((source_id, evidence_index))
            query_started = time.perf_counter()
            predictions = model.predict(
                pairs,
                batch_size=args.batch_size,
                show_progress_bar=False,
                prompt_name="legal",
            ) if pairs else []
            best_scores: dict[str, float] = {}
            for score, (source_id, _) in zip(predictions, locations):
                score = float(score)
                if score > best_scores.get(source_id, float("-inf")):
                    best_scores[source_id] = score
            cached_scores[query_id] = [
                {
                    "id": source_id,
                    "score": best_scores.get(source_id),
                    "candidate_rank": rank,
                    "evidence": [item["source"] for item in evidence[source_id]],
                }
                for rank, source_id in enumerate(candidate_ids, start=1)
            ]
            runtime_rows[query_id] = {
                "cached": False,
                "candidate_count": len(candidate_ids),
                "matched_candidate_count": sum(bool(evidence[source_id]) for source_id in candidate_ids),
                "evidence_pair_count": len(pairs),
                "seconds": round(time.perf_counter() - query_started, 3),
                "max_rss_mb": max_rss_mb(),
            }
            atomic_dump(args.scores_output, cached_scores)
            print(
                f"[{index}/{len(query_items)}] {query_id}: {len(pairs)} pairs, "
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
        "model": args.model,
        "instruction": args.instruction,
        "device": device,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "queries": len(query_items),
        "total_seconds": round(time.perf_counter() - started, 3),
        "max_rss_mb": max(float(previous_stats.get("max_rss_mb", 0.0)), max_rss_mb()),
        "swap_usage": swap_usage(),
        "per_query": runtime_rows,
    }
    atomic_dump(args.stats_output, stats)
    print(f"wrote {len(direct)} direct top-{args.top_k} predictions to {args.output}")
    print(f"max RSS: {stats['max_rss_mb']} MB; swap: {stats['swap_usage']}")


if __name__ == "__main__":
    main()
