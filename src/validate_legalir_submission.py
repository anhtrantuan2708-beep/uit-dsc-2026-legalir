#!/usr/bin/env python3
"""Validate a LegalIR submission.json against a query file."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    errors: list[str] = []

    if not isinstance(submission, dict):
        errors.append("submission root must be a JSON object")
    else:
        missing = set(queries) - set(submission)
        extra = set(submission) - set(queries)
        if missing:
            errors.append(f"missing query IDs: {len(missing)}")
        if extra:
            errors.append(f"unknown query IDs: {len(extra)}")

        for query_id, row in submission.items():
            if not isinstance(row, dict) or not isinstance(row.get("answer"), list):
                errors.append(f"{query_id}: expected object with answer list")
                continue
            answers = row["answer"]
            if len(answers) > 5:
                errors.append(f"{query_id}: {len(answers)} answers; maximum is 5")
            if any(not isinstance(item, (str, int)) for item in answers):
                errors.append(f"{query_id}: answer IDs must be strings or integers")
            if len(set(map(str, answers))) != len(answers):
                errors.append(f"{query_id}: duplicate answer IDs")

    if errors:
        print("INVALID")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    print(f"VALID: {len(submission)} query records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
