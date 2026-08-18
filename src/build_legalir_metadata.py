#!/usr/bin/env python3
"""Derive searchable document metadata from the official LegalIR corpus only.

The competition corpus is intentionally kept unchanged.  This script creates a
local, ignored sidecar that lets retrieval search document number, type, title
and issuer separately from a very long legal passage.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


NUMBER_RE = re.compile(r"(?im)\bSố\s*[:：]?\s*([A-ZĐ0-9][A-ZĐ0-9./_-]{2,60})")
TYPE_RE = re.compile(
    r"\b(Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Chỉ\s+thị|Nghị\s+quyết|Pháp\s+lệnh)\b",
    re.IGNORECASE,
)


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip()


def infer_title(passage: str) -> str:
    """Use an all-caps heading near the official document header when name is absent."""
    candidates = []
    for line in passage[:8000].splitlines():
        value = clean(line)
        letters = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", value)
        if 12 <= len(value) <= 220 and len(letters) >= 8 and value == value.upper():
            if not any(skip in value for skip in ("CỘNG HÒA", "ĐỘC LẬP", "TỰ DO", "HẠNH PHÚC", "-------")):
                candidates.append(value)
    return max(candidates, key=len, default="")


def infer_issuer(passage: str) -> str:
    header = passage[:2500]
    before_number = header.split("Số", 1)[0]
    lines = [clean(line) for line in before_number.splitlines()]
    candidates = [line for line in lines if 5 <= len(line) <= 180 and "CỘNG HÒA" not in line.upper()]
    return " ".join(candidates[:3])


def extract(row: dict) -> dict:
    passage = str(row.get("passage", ""))
    raw_name = clean(str(row.get("name", "")))
    number = NUMBER_RE.search(passage[:8000])
    kind = TYPE_RE.search((raw_name + "\n" + passage[:8000]))
    title = raw_name or infer_title(passage)
    document_type = clean(kind.group(1)).lower() if kind else ""
    issuer = infer_issuer(passage)
    metadata = " ".join(part for part in (document_type, number.group(1) if number else "", title, issuer) if part)
    return {
        "id": str(row["id"]),
        "document_number": number.group(1) if number else "",
        "document_type": document_type,
        "title": title,
        "issuer": issuer,
        "search_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metadata = {}
    type_counts = Counter()
    coverage = Counter()
    for file in sorted(args.corpus.glob("*.json")):
        item = extract(json.loads(file.read_text(encoding="utf-8")))
        metadata[item["id"]] = item
        for field in ("document_number", "document_type", "title", "issuer"):
            coverage[field] += bool(item[field])
        type_counts[item["document_type"] or "unknown"] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(f"wrote metadata for {len(metadata)} documents to {args.output}")
    print("coverage:", dict(coverage))
    print("types:", dict(type_counts.most_common()))


if __name__ == "__main__":
    main()
