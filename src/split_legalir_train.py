#!/usr/bin/env python3
"""Create deterministic train/dev query splits for LegalIR experiments."""

import argparse
import hashlib
import json
from pathlib import Path


def belongs_to_dev(query_id: str, ratio: int) -> bool:
    digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % ratio == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("train_output", type=Path)
    parser.add_argument("dev_output", type=Path)
    parser.add_argument("--dev-ratio", type=int, default=7)
    args = parser.parse_args()
    if args.dev_ratio < 2:
        raise SystemExit("--dev-ratio must be at least 2")

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    train = {}
    dev = {}
    for query_id, row in rows.items():
        target = dev if belongs_to_dev(query_id, args.dev_ratio) else train
        target[query_id] = row

    for path, payload in [(args.train_output, train), (args.dev_output, dev)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train queries: {len(train)}")
    print(f"dev queries: {len(dev)}")


if __name__ == "__main__":
    main()
