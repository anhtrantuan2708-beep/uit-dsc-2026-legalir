#!/usr/bin/env python3
"""Retrieve source documents through a disk-backed SQLite FTS5 chunk index."""

import argparse
import json
import re
import sqlite3
from pathlib import Path

from chunk_legalir_contexts import split_document
from legalir_baseline import STOPWORDS

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def query_expression(text: str) -> str:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in STOPWORDS or token in tokens:
            continue
        tokens.append(token)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def build_index(corpus: Path, database: Path, chunk_size: int, overlap: int) -> None:
    if database.exists():
        connection = sqlite3.connect(database)
        try:
            ready = connection.execute(
                "SELECT value FROM metadata WHERE key='ready'"
            ).fetchone()
            if ready and ready[0] == "1":
                print(f"reusing completed FTS index: {database}")
                connection.close()
                return
        except sqlite3.Error:
            pass
        connection.close()
        database.unlink()

    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5("
        "source_id UNINDEXED, name, passage, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    batch = []
    chunk_count = 0
    for file in sorted(corpus.glob("*.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        source_id = str(row["id"])
        name = str(row.get("name", "")).replace("-", " ")
        for passage in split_document(str(row["passage"]), chunk_size, overlap):
            batch.append((source_id, name, passage))
            chunk_count += 1
            if len(batch) >= 1000:
                connection.executemany(
                    "INSERT INTO chunks(source_id, name, passage) VALUES (?, ?, ?)", batch
                )
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO chunks(source_id, name, passage) VALUES (?, ?, ?)", batch
        )
    connection.execute("INSERT INTO metadata(key, value) VALUES ('ready', '1')")
    connection.commit()
    connection.close()
    print(f"indexed {chunk_count} chunks in {database}")


def retrieve(queries: Path, database: Path, output: Path, chunk_k: int, top_k: int) -> None:
    rows = json.loads(queries.read_text(encoding="utf-8"))
    connection = sqlite3.connect(database)
    result = {}
    for query_id, row in rows.items():
        expression = query_expression(str(row["question"]))
        ranked_documents = []
        seen = set()
        if expression:
            matches = connection.execute(
                "SELECT source_id, bm25(chunks, 0.0, 2.0, 1.0) AS score "
                "FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
                (expression, chunk_k),
            )
            for source_id, _ in matches:
                source_id = str(source_id)
                if source_id not in seen:
                    ranked_documents.append(source_id)
                    seen.add(source_id)
                if len(ranked_documents) == top_k:
                    break
        result[str(query_id)] = {"answer": ranked_documents}
    connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} predictions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--chunk-k", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    build_index(args.corpus, args.database, args.chunk_size, args.overlap)
    retrieve(args.queries, args.database, args.output, args.chunk_k, args.top_k)


if __name__ == "__main__":
    main()
