#!/usr/bin/env python3
"""Retrieve LegalIR source IDs from hierarchy-aware child records with SQLite FTS5.

This is deliberately independent from the existing flat FTS5 index.  It accepts
the JSONL produced by build_hierarchical_legalir_corpus.py, ranks matching
children, and aggregates them back to the submission-level source document ID.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from legalir_baseline import STOPWORDS


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
SCHEMA_VERSION = "hierarchy-fts-v1"


def query_expression(text: str) -> str:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in STOPWORDS or token in tokens:
            continue
        tokens.append(token)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def source_signature(children_path: Path) -> dict[str, str]:
    stat = children_path.stat()
    return {
        "schema": SCHEMA_VERSION,
        "children_path": str(children_path.resolve()),
        "children_size": str(stat.st_size),
        "children_mtime_ns": str(stat.st_mtime_ns),
    }


def index_is_current(database: Path, signature: dict[str, str]) -> bool:
    if not database.exists():
        return False
    try:
        connection = sqlite3.connect(database)
        rows = dict(connection.execute("SELECT key, value FROM metadata"))
        connection.close()
        return all(rows.get(key) == value for key, value in signature.items()) and rows.get("ready") == "1"
    except sqlite3.Error:
        return False


def build_index(children_path: Path, database: Path) -> None:
    signature = source_signature(children_path)
    if index_is_current(database, signature):
        print(f"reusing completed hierarchy FTS index: {database}")
        return

    if database.exists():
        database.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "CREATE VIRTUAL TABLE children USING fts5("
        "child_id UNINDEXED, source_id UNINDEXED, parent_id UNINDEXED, path, body, "
        "tokenize='unicode61 remove_diacritics 2')"
    )

    batch: list[tuple[str, str, str, str, str]] = []
    count = 0
    with children_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            passage = str(row["passage"])
            path, separator, body = passage.partition("\n")
            if not separator:
                path = str(row.get("path", ""))
                body = passage
            batch.append((
                str(row["id"]),
                str(row["source_id"]),
                str(row["parent_id"]),
                path,
                body,
            ))
            count += 1
            if len(batch) >= 2_000:
                connection.executemany(
                    "INSERT INTO children(child_id, source_id, parent_id, path, body) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO children(child_id, source_id, parent_id, path, body) VALUES (?, ?, ?, ?, ?)", batch
        )
    connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", signature.items())
    connection.execute("INSERT INTO metadata(key, value) VALUES ('ready', '1')")
    connection.commit()
    connection.close()
    print(f"indexed {count} hierarchy children in {database}")


def aggregate(scores: list[float], method: str) -> float:
    scores = sorted(scores, reverse=True)
    if method == "maxp":
        return scores[0]
    if method == "top2sum":
        # One supporting child is useful, but cannot completely outweigh the best match.
        return scores[0] + (0.25 * scores[1] if len(scores) > 1 else 0.0)
    if method == "logsumexp":
        selected = scores[:3]
        maximum = selected[0]
        return maximum + math.log(sum(math.exp(score - maximum) for score in selected))
    raise ValueError(f"unknown aggregation method: {method}")


def retrieve(
    queries_path: Path,
    database: Path,
    output_path: Path,
    scores_path: Path | None,
    chunk_k: int,
    top_k: int,
    aggregation: str,
) -> None:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(database)
    predictions: dict[str, dict[str, list[str]]] = {}
    score_rows: dict[str, list[dict[str, object]]] = {}

    for query_id, row in queries.items():
        expression = query_expression(str(row["question"]))
        per_source: dict[str, list[tuple[float, str]]] = defaultdict(list)
        if expression:
            matches = connection.execute(
                "SELECT source_id, child_id, -bm25(children, 0.0, 0.0, 0.0, 0.35, 1.0) AS score "
                "FROM children WHERE children MATCH ? ORDER BY score DESC LIMIT ?",
                (expression, chunk_k),
            )
            for source_id, child_id, score in matches:
                per_source[str(source_id)].append((float(score), str(child_id)))

        ranked = []
        for source_id, pairs in per_source.items():
            ordered = sorted(pairs, reverse=True)
            ranked.append((aggregate([score for score, _ in ordered], aggregation), source_id, ordered[:3]))
        ranked.sort(reverse=True)
        top = ranked[:top_k]
        predictions[str(query_id)] = {"answer": [source_id for _, source_id, _ in top]}
        if scores_path:
            score_rows[str(query_id)] = [
                {
                    "source_id": source_id,
                    "score": round(score, 8),
                    "child_ids": [child_id for _, child_id in children],
                }
                for score, source_id, children in top
            ]

    connection.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    if scores_path:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("children", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--chunk-k", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--aggregate", choices=("maxp", "top2sum", "logsumexp"), default="maxp")
    args = parser.parse_args()
    if args.top_k < 1 or args.chunk_k < args.top_k:
        raise SystemExit("chunk-k must be at least top-k and both must be positive")

    build_index(args.children, args.database)
    retrieve(
        args.queries,
        args.database,
        args.output,
        args.scores_output,
        args.chunk_k,
        args.top_k,
        args.aggregate,
    )


if __name__ == "__main__":
    main()
