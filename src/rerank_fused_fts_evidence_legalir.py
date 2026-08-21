#!/usr/bin/env python3
"""Rerank fused LegalIR candidates using their matched evidence from two FTS indexes.

For a candidate document, flat FTS and hierarchy FTS may surface different
passages.  We score both passages with the same cross-encoder and retain the
document's best evidence score.  This is a multi-evidence rerank, not a wider
candidate search: it never introduces IDs outside the supplied candidate list.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from fts_chunk_legalir import query_expression


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flat_evidence(connection, expression: str, wanted: set[str], chunk_k: int) -> dict[str, str]:
    found = {}
    if not expression or not wanted:
        return found
    rows = connection.execute(
        "SELECT source_id, name, passage FROM chunks WHERE chunks MATCH ? "
        "ORDER BY bm25(chunks, 0.0, 2.0, 1.0) LIMIT ?",
        (expression, chunk_k),
    )
    for source_id, name, passage in rows:
        source_id = str(source_id)
        if source_id in wanted and source_id not in found:
            found[source_id] = f"{name}\n{passage}".strip()
            if len(found) == len(wanted):
                break
    return found


def hierarchy_evidence(connection, expression: str, wanted: set[str], chunk_k: int) -> dict[str, str]:
    found = {}
    if not expression or not wanted:
        return found
    rows = connection.execute(
        "SELECT source_id, path, body FROM children WHERE children MATCH ? "
        "ORDER BY bm25(children, 0.0, 0.0, 0.0, 0.35, 1.0) LIMIT ?",
        (expression, chunk_k),
    )
    for source_id, path, body in rows:
        source_id = str(source_id)
        if source_id in wanted and source_id not in found:
            found[source_id] = f"{path}\n{body}".strip()
            if len(found) == len(wanted):
                break
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("flat_database", type=Path)
    parser.add_argument("hierarchy_database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--chunk-k", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model-cache", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    queries, candidate_rows = load(args.queries), load(args.candidates)
    query_items = list(queries.items())[args.query_start:]
    if args.max_queries is not None:
        query_items = query_items[: args.max_queries]
    flat_connection = sqlite3.connect(args.flat_database)
    hierarchy_connection = sqlite3.connect(args.hierarchy_database)
    pairs, locations, candidate_ranks = [], [], {}
    evidence_counts = defaultdict(int)
    for query_id, row in query_items:
        query_id = str(query_id)
        candidate_ids = [str(item) for item in candidate_rows.get(query_id, {}).get("answer", [])[: args.candidate_k]]
        candidate_ranks[query_id] = {source_id: rank for rank, source_id in enumerate(candidate_ids, start=1)}
        wanted = set(candidate_ids)
        expression = query_expression(str(row["question"]))
        sources = (
            flat_evidence(flat_connection, expression, wanted, args.chunk_k),
            hierarchy_evidence(hierarchy_connection, expression, wanted, args.chunk_k),
        )
        seen = set()
        for source_name, evidence_map in zip(("flat", "hierarchy"), sources):
            for source_id, evidence in evidence_map.items():
                key = (source_id, evidence)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append([str(row["question"]), evidence])
                locations.append((query_id, source_id, source_name))
                evidence_counts[source_name] += 1
    flat_connection.close()
    hierarchy_connection.close()

    print(f"scoring {len(pairs)} matched-evidence pairs: {dict(evidence_counts)}")
    model = CrossEncoder(args.model, cache_folder=str(args.model_cache))
    predictions = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True) if pairs else []
    best = defaultdict(dict)
    provenance = defaultdict(dict)
    for score, (query_id, source_id, source_name) in zip(predictions, locations):
        score = float(score)
        if score > best[query_id].get(source_id, float("-inf")):
            best[query_id][source_id] = score
            provenance[query_id][source_id] = source_name

    ranking, score_rows = {}, {}
    for query_id, _ in query_items:
        query_id = str(query_id)
        ranked = sorted(
            best[query_id].items(),
            key=lambda item: (-item[1], candidate_ranks[query_id].get(item[0], 10**9), item[0]),
        )
        ranking[query_id] = {"answer": [source_id for source_id, _ in ranked[: args.top_k]]}
        score_rows[query_id] = [
            {
                "id": source_id,
                "score": score,
                "candidate_rank": candidate_ranks[query_id].get(source_id),
                "evidence_source": provenance[query_id][source_id],
            }
            for source_id, score in ranked
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(ranking)} reranked query rows")


if __name__ == "__main__":
    main()
