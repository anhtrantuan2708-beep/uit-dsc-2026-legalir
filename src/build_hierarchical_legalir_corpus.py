#!/usr/bin/env python3
"""Build legal-aware child and parent records from the LegalIR corpus.

The source corpus contains long Vietnamese legal documents.  This script keeps
the document ID unchanged, but creates two views:

* children: short Clause/Point-sized retrieval records with a legal path;
* parents: Article-sized evidence windows for a later reranker.

No external data or generated metadata is used.  The parser is intentionally
rule-based so that its coverage and mistakes are easy to audit before it is
used by a retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CHAPTER_RE = re.compile(r"^\s*(CHƯƠNG\s+(?:[IVXLCDM]+|\d+)\b.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(MỤC\s+(?:[IVXLCDM]+|\d+)\b.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*(Điều\s+\d+[a-zA-Z]?\b.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*(?:(Khoản)\s+)?(\d+)\.\s*(.*)$", re.IGNORECASE)
POINT_RE = re.compile(r"^\s*(?:(Điểm)\s+)?([a-zđ])\)\s*(.*)$", re.IGNORECASE)
DOC_NUMBER_RE = re.compile(
    r"\b(?:Số\s*:\s*)?([0-9]{1,4}(?:/[0-9]{2,4})?(?:/[A-ZĐ\-]{1,24}){1,3})\b",
    re.IGNORECASE,
)


@dataclass
class Article:
    number: int
    chapter: str
    section: str
    header: str
    lines: list[str]


def normalize_lines(text: str) -> list[str]:
    """Normalize spacing but retain the source's logical line boundaries."""
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def short_title(name: str) -> str:
    title = re.sub(r"[-_]+", " ", name)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:240]


def document_label(row: dict[str, object], metadata_row: dict[str, object] | None, text: str) -> str:
    """Create a short lexical path without changing the organizer document ID.

    Raw `name` is blank for many documents.  The local metadata sidecar was
    derived only from the same organizer corpus and gives a compact canonical
    title; it is useful as a low-weight retrieval field, not a new label.
    """
    number = document_number(text, metadata_row)
    raw_title = short_title(str(row.get("name", "")))
    metadata_title = short_title(str((metadata_row or {}).get("title", "")))
    titles = []
    for title in (raw_title, metadata_title):
        if title and title.casefold() not in {item.casefold() for item in titles}:
            titles.append(title)
    return " ".join(part for part in (number, *titles) if part).strip()


def document_number(text: str, metadata_row: dict[str, object] | None) -> str:
    if metadata_row:
        candidate = str(metadata_row.get("document_number", "")).strip()
        if re.fullmatch(r"[0-9]{1,4}(?:/[0-9]{2,4})?(?:/[A-Za-zĐđ\-]{1,24}){1,3}", candidate):
            return candidate.upper()
    match = DOC_NUMBER_RE.search(text[:5000])
    return match.group(1).upper() if match else ""


def split_to_limit(text: str, limit: int, overlap: int = 60) -> list[str]:
    """Split long content at a nearby space, retaining a small context overlap."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    pieces = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + limit // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def article_windows(text: str, limit: int, overlap: int = 300) -> list[tuple[int, int, str]]:
    """Return evidence windows and their offsets inside an Article."""
    if len(text) <= limit:
        return [(0, len(text), text)]
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + limit // 2:
                end = boundary + 1
        windows.append((start, end, text[start:end].strip()))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return windows


def parse_articles(lines: list[str]) -> tuple[list[Article], list[str], Counter[str]]:
    """Extract Article spans while carrying the latest Chapter and Section."""
    articles: list[Article] = []
    preamble: list[str] = []
    current: Article | None = None
    chapter = ""
    section = ""
    seen = Counter()

    for line in lines:
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter = chapter_match.group(1)
            section = ""
            seen["chapter"] += 1
            if not current:
                preamble.append(line)
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1)
            seen["section"] += 1
            if not current:
                preamble.append(line)
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            current = Article(
                number=len(articles) + 1,
                chapter=chapter,
                section=section,
                header=article_match.group(1),
                lines=[article_match.group(1)],
            )
            articles.append(current)
            seen["article"] += 1
            continue

        if current:
            current.lines.append(line)
        else:
            preamble.append(line)

    return articles, preamble, seen


def split_article_units(article: Article) -> list[tuple[str, str]]:
    """Return (unit_label, body) pairs for clauses and points inside one Article."""
    body_lines = article.lines[1:]
    units: list[tuple[str, list[str]]] = []
    current_label = "Nội dung Điều"
    current_lines: list[str] = []

    for line in body_lines:
        point_match = POINT_RE.match(line)
        clause_match = CLAUSE_RE.match(line) if not point_match else None
        if point_match:
            if current_lines:
                units.append((current_label, current_lines))
            current_label = f"Điểm {point_match.group(2)}"
            current_lines = [line]
        elif clause_match:
            if current_lines:
                units.append((current_label, current_lines))
            current_label = f"Khoản {clause_match.group(2)}"
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        units.append((current_label, current_lines))
    if not units:
        units = [("Nội dung Điều", [article.header])]
    return [(label, " ".join(lines).strip()) for label, lines in units if " ".join(lines).strip()]


def choose_parent_id(windows: list[tuple[int, int, str]], offset: int, parent_prefix: str) -> str:
    for index, (start, end, _) in enumerate(windows):
        if start <= offset < end:
            return f"{parent_prefix}__evidence_{index}"
    return f"{parent_prefix}__evidence_0"


def record_path(doc_prefix: str, article: Article | None, unit_label: str = "") -> str:
    parts = [doc_prefix]
    if article:
        parts.extend(part for part in (article.chapter, article.section, article.header, unit_label) if part)
    return " | ".join(parts)


def load_metadata(path: Path | None) -> dict[str, dict[str, object]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--child-chars", type=int, default=450)
    parser.add_argument("--parent-chars", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="audit only the first N documents")
    parser.add_argument("--tag", default="", help="suffix for a non-production smoke output")
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    if args.child_chars < 100 or args.parent_chars < args.child_chars:
        raise SystemExit("child-chars must be >=100 and parent-chars must be >= child-chars")

    suffix = f"_{args.tag.strip()}" if args.tag.strip() else ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_path = args.output_dir / f"legalir_hierarchy_children{suffix}.jsonl"
    parent_path = args.output_dir / f"legalir_hierarchy_parents{suffix}.jsonl"
    metadata = load_metadata(args.metadata)
    files = sorted(args.corpus.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    examples: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    fingerprint_sample_limit = 200_000
    child_count = 0
    parent_count = 0

    with child_path.open("w", encoding="utf-8") as child_handle, parent_path.open("w", encoding="utf-8") as parent_handle:
        for file in files:
            row = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or "id" not in row or "passage" not in row:
                stats["invalid_documents"] += 1
                continue

            source_id = str(row["id"])
            passage = str(row["passage"])
            doc_label = document_label(row, metadata.get(source_id), passage) or f"Document {source_id}"
            lines = normalize_lines(passage)
            articles, preamble, seen = parse_articles(lines)
            stats["documents"] += 1
            stats["documents_with_article"] += int(bool(articles))
            stats["documents_with_chapter"] += int(bool(seen["chapter"]))
            stats["documents_with_section"] += int(bool(seen["section"]))

            if preamble:
                preamble_text = " ".join(preamble)
                parent_id = f"{source_id}__preamble__evidence_0"
                parent_handle.write(json.dumps({
                    "id": parent_id,
                    "source_id": source_id,
                    "path": doc_label,
                    "passage": f"{doc_label}\n{preamble_text[: args.parent_chars]}",
                    "kind": "preamble",
                }, ensure_ascii=False) + "\n")
                parent_count += 1
                for piece_index, piece in enumerate(split_to_limit(preamble_text, args.child_chars)):
                    child_id = f"{source_id}__preamble__child_{piece_index}"
                    child = {
                        "id": child_id,
                        "source_id": source_id,
                        "parent_id": parent_id,
                        "path": doc_label,
                        "passage": f"{doc_label}\n{piece}",
                        "kind": "preamble",
                    }
                    child_handle.write(json.dumps(child, ensure_ascii=False) + "\n")
                    child_count += 1
                    if child_count <= fingerprint_sample_limit:
                        fingerprints[hashlib.sha1(str(child["passage"]).encode("utf-8")).hexdigest()] += 1

            for article in articles:
                article_text = " ".join(article.lines).strip()
                parent_prefix = f"{source_id}__article_{article.number}"
                path = record_path(doc_label, article)
                windows = article_windows(article_text, args.parent_chars)
                for evidence_index, (_, _, evidence) in enumerate(windows):
                    parent_handle.write(json.dumps({
                        "id": f"{parent_prefix}__evidence_{evidence_index}",
                        "source_id": source_id,
                        "path": path,
                        "passage": f"{path}\n{evidence}",
                        "kind": "article_evidence",
                    }, ensure_ascii=False) + "\n")
                    parent_count += 1

                for unit_index, (unit_label, unit_text) in enumerate(split_article_units(article)):
                    stats["clause_or_point_units"] += int(unit_label != "Nội dung Điều")
                    offset = article_text.find(unit_text)
                    parent_id = choose_parent_id(windows, max(offset, 0), parent_prefix)
                    unit_path = record_path(doc_label, article, unit_label)
                    for piece_index, piece in enumerate(split_to_limit(unit_text, args.child_chars)):
                        child_id = f"{parent_prefix}__unit_{unit_index}__child_{piece_index}"
                        child = {
                            "id": child_id,
                            "source_id": source_id,
                            "parent_id": parent_id,
                            "path": unit_path,
                            "passage": f"{unit_path}\n{piece}",
                            "kind": "legal_unit",
                        }
                        child_handle.write(json.dumps(child, ensure_ascii=False) + "\n")
                        child_count += 1
                        if child_count <= fingerprint_sample_limit:
                            fingerprints[hashlib.sha1(str(child["passage"]).encode("utf-8")).hexdigest()] += 1
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "source_id": source_id,
                                    "path": unit_path,
                                    "passage": str(child["passage"])[:500],
                                }
                            )

    duplicate_count = sum(count - 1 for count in fingerprints.values() if count > 1)
    audit = {
        "corpus": str(args.corpus),
        "metadata": str(args.metadata) if args.metadata else None,
        "documents_seen": stats["documents"],
        "documents_with_article": stats["documents_with_article"],
        "documents_with_chapter": stats["documents_with_chapter"],
        "documents_with_section": stats["documents_with_section"],
        "article_coverage": stats["documents_with_article"] / stats["documents"] if stats["documents"] else 0.0,
        "child_records": child_count,
        "parent_records": parent_count,
        "clause_or_point_units": stats["clause_or_point_units"],
        "duplicate_child_passages_in_first_200k": duplicate_count,
        "duplicate_scan_records": min(child_count, fingerprint_sample_limit),
        "invalid_documents": stats["invalid_documents"],
        "child_chars": args.child_chars,
        "parent_chars": args.parent_chars,
        "examples": examples,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in audit if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
