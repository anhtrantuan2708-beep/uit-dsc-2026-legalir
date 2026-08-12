#!/usr/bin/env python3
"""Build one compact, evenly sampled representation for every legal document.

Dense encoders have an input limit.  Rather than embed only the first part of a
very long statute, this keeps its title and samples passages across its full
length.  The result still has one row (and one submission ID) per document.
"""

import argparse
import json
from pathlib import Path


def representative_text(text: str, budget: int, slices: int) -> str:
    text = " ".join(text.split())
    if len(text) <= budget:
        return text

    slice_size = max(250, budget // slices)
    starts = [round((len(text) - slice_size) * index / (slices - 1)) for index in range(slices)]
    parts = [text[start : start + slice_size] for start in starts]
    return "\n...\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--slices", type=int, default=6)
    args = parser.parse_args()
    if args.slices < 2:
        raise SystemExit("slices must be at least 2")

    records = []
    for file in sorted(args.corpus.glob("*.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or "id" not in row or "passage" not in row:
            continue
        records.append(
            {
                "id": str(row["id"]),
                "name": row.get("name", ""),
                "passage": representative_text(str(row["passage"]), args.budget, args.slices),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} representative documents to {args.output}")


if __name__ == "__main__":
    main()
