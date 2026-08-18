#!/usr/bin/env python3
"""Split LegalIR documents into retrieval chunks while preserving source IDs."""

import argparse
import json
import re
from pathlib import Path

ARTICLE_BOUNDARY = re.compile(r"(?=\b(?:Điều|Chương|Mục)\s+(?:\d+|[IVXLCDM]+)\b)", re.IGNORECASE)
CLAUSE_BOUNDARY = re.compile(r"(?=^\s*(?:(?:Khoản\s+)?\d+\.\s+|(?:Điểm\s+)?[a-zđ]\)\s+))", re.IGNORECASE | re.MULTILINE)
ARTICLE_HEADER = re.compile(r"(?im)^\s*(Điều\s+\d+[^\n]*)")


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    result = []
    start = 0
    while start < len(text):
        result.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return result


def split_document(text: str, size: int, overlap: int) -> list[str]:
    sections = [part.strip() for part in ARTICLE_BOUNDARY.split(text) if part.strip()]
    chunks = []
    for section in sections or [text]:
        chunks.extend(fixed_chunks(section, size, overlap))
    return chunks


def split_document_structured(text: str, size: int, overlap: int) -> list[str]:
    """Split long articles by clauses, repeating the article heading as context."""
    chunks = []
    for section in [part.strip() for part in ARTICLE_BOUNDARY.split(text) if part.strip()] or [text]:
        header_match = ARTICLE_HEADER.search(section)
        header = header_match.group(1).strip() if header_match else ""
        parts = [part.strip() for part in CLAUSE_BOUNDARY.split(section) if part.strip()]
        for part in parts or [section]:
            contextual = part if not header or part.startswith(header) else f"{header}\n{part}"
            chunks.extend(fixed_chunks(contextual, size, overlap))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--structured", action="store_true", help="split long articles at legal clause/point boundaries")
    args = parser.parse_args()
    if args.overlap >= args.chunk_size:
        raise SystemExit("overlap must be smaller than chunk-size")

    chunks = []
    for file in sorted(args.corpus.glob("*.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or "id" not in row or "passage" not in row:
            continue
        source_id = str(row["id"])
        splitter = split_document_structured if args.structured else split_document
        for number, passage in enumerate(splitter(str(row["passage"]), args.chunk_size, args.overlap)):
            chunks.append(
                {
                    "id": f"{source_id}__chunk_{number}",
                    "source_id": source_id,
                    "name": row.get("name", ""),
                    "passage": passage,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"documents: {len(list(args.corpus.glob('*.json')))}")
    print(f"chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
