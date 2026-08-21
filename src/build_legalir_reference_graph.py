#!/usr/bin/env python3
"""Build a one-hop reference graph from the organizer-provided LegalIR corpus.

The graph is intentionally conservative: an edge is created only when a legal
document number found in text resolves to exactly one source ID in the local
metadata.  It is an oracle/candidate-expansion input, not a claim that every
cited document answers the query.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DOC_NUMBER_RE = re.compile(
    r"\b(?:số\s*)?([0-9]{1,4}(?:/[0-9]{2,4})?(?:/[A-ZĐ\-]{1,24}){1,3})\b",
    re.IGNORECASE,
)
RELATION_PATTERNS = (
    ("AMENDS", re.compile(r"sửa đổi|bổ sung|thay thế|bãi bỏ", re.IGNORECASE)),
    ("GUIDES", re.compile(r"hướng dẫn|quy định chi tiết|thi hành", re.IGNORECASE)),
    ("PURSUANT_TO", re.compile(r"căn cứ|chiếu theo", re.IGNORECASE)),
)


def normalize_number(value: str) -> str:
    return re.sub(r"\s+", "", value).upper().replace("Đ", "D")


def infer_relation(text: str, start: int, end: int) -> str:
    context = text[max(0, start - 100) : min(len(text), end + 100)]
    for label, pattern in RELATION_PATTERNS:
        if pattern.search(context):
            return label
    return "REFERS_TO"


def metadata_number_map(metadata: dict[str, dict[str, object]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for source_id, row in metadata.items():
        number = str(row.get("document_number", "")).strip()
        if DOC_NUMBER_RE.fullmatch(number):
            mapping[normalize_number(number)].add(str(source_id))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    number_to_ids = metadata_number_map(metadata)
    files = sorted(args.corpus.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    edges: dict[str, dict[tuple[str, str], dict[str, object]]] = defaultdict(dict)
    relation_counts: Counter[str] = Counter()
    stats = Counter()
    collisions: Counter[str] = Counter()

    for file in files:
        row = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or "id" not in row or "passage" not in row:
            stats["invalid_documents"] += 1
            continue
        source_id = str(row["id"])
        text = str(row["passage"])
        stats["documents"] += 1
        for match in DOC_NUMBER_RE.finditer(text):
            stats["references_found"] += 1
            reference = normalize_number(match.group(1))
            targets = number_to_ids.get(reference, set())
            if len(targets) == 1:
                target_id = next(iter(targets))
                if target_id == source_id:
                    stats["self_references"] += 1
                    continue
                relation = infer_relation(text, match.start(), match.end())
                key = (target_id, relation)
                existing = edges[source_id].get(key)
                if existing:
                    existing["count"] = int(existing["count"]) + 1
                else:
                    edges[source_id][key] = {
                        "target_id": target_id,
                        "relation": relation,
                        "reference": reference,
                        "count": 1,
                    }
                    relation_counts[relation] += 1
                stats["references_resolved"] += 1
            elif len(targets) > 1:
                collisions[reference] += 1
                stats["references_ambiguous"] += 1
            else:
                stats["references_unresolved"] += 1

    graph = {
        source_id: sorted(items.values(), key=lambda item: (-int(item["count"]), str(item["target_id"])))
        for source_id, items in edges.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    resolved = stats["references_resolved"]
    found = stats["references_found"]
    audit = {
        "documents_seen": stats["documents"],
        "invalid_documents": stats["invalid_documents"],
        "metadata_numbers": len(number_to_ids),
        "metadata_number_collisions": sum(1 for values in number_to_ids.values() if len(values) > 1),
        "references_found": found,
        "references_resolved": resolved,
        "references_unresolved": stats["references_unresolved"],
        "references_ambiguous": stats["references_ambiguous"],
        "self_references": stats["self_references"],
        "resolution_rate": resolved / found if found else 0.0,
        "source_nodes_with_edges": len(graph),
        "edges": sum(len(items) for items in graph.values()),
        "relations": dict(relation_counts),
        "top_ambiguous_numbers": collisions.most_common(20),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
