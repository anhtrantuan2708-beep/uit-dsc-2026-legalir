#!/usr/bin/env python3
"""Retrieve LegalIR document IDs from a compact metadata-only FTS index.

Unlike the chunk index, metadata is stored once per document.  It is a cheap
supplementary candidate source for document numbers, legal type and issuer.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from fts_chunk_legalir import query_expression


def build(metadata_path: Path, database: Path) -> None:
    if database.exists():
        return
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE VIRTUAL TABLE documents USING fts5("
        "source_id UNINDEXED, document_number, document_type, title, issuer, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    rows = [
        (document_id, item.get("document_number", ""), item.get("document_type", ""), item.get("title", ""), item.get("issuer", ""))
        for document_id, item in records.items()
    ]
    connection.executemany(
        "INSERT INTO documents(source_id, document_number, document_type, title, issuer) VALUES (?, ?, ?, ?, ?)", rows
    )
    connection.commit()
    connection.close()
    print(f"indexed metadata for {len(rows)} documents in {database}")


def retrieve(queries_path: Path, database: Path, output: Path, top_k: int) -> None:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(database)
    result = {}
    for query_id, row in queries.items():
        expression = query_expression(str(row["question"]))
        if not expression:
            result[str(query_id)] = {"answer": []}
            continue
        matches = connection.execute(
            "SELECT source_id FROM documents WHERE documents MATCH ? "
            "ORDER BY bm25(documents, 0.0, 6.0, 3.0, 3.0, 1.0) LIMIT ?",
            (expression, top_k),
        )
        result[str(query_id)] = {"answer": [str(source_id) for (source_id,) in matches]}
    connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} metadata rankings to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    build(args.metadata, args.database)
    retrieve(args.queries, args.database, args.output, args.top_k)


if __name__ == "__main__":
    main()
