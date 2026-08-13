#!/usr/bin/env python3
"""Report LegalIR recall by simple, interpretable Vietnamese query slices."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PATTERNS = {
    "article_clause": r"\b(?:điều|khoản|điểm)\s+\d+",
    "legal_document": r"\b(?:luật|nghị định|thông tư|quyết định|bộ luật)\b",
    "document_number": r"\b\d+[/-]\d{2,4}(?:/|[-–])?[a-zđ]+",
    "duration": r"\b(?:bao lâu|thời hạn|bao nhiêu ngày|bao nhiêu tháng|bao nhiêu năm)\b",
    "amount": r"\b(?:bao nhiêu tiền|mức phạt|phạt tiền|tỷ lệ|phần trăm)\b",
    "procedure": r"\b(?:thủ tục|hồ sơ|đăng ký|cấp phép|trình tự)\b",
    "condition": r"\b(?:điều kiện|trường hợp nào|khi nào|được phép)\b",
    "authority": r"\b(?:cơ quan nào|ai có thẩm quyền|thẩm quyền)\b",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    gold, predictions = load(args.gold), load(args.predictions)
    stats = defaultdict(lambda: [0, 0.0, 0])

    for query_id, row in gold.items():
        text = row.get("question", "").lower()
        gold_ids = {str(value) for value in row.get("answer", [])}
        predicted = {
            str(value)
            for value in predictions.get(query_id, {}).get("answer", [])[: args.top_k]
        }
        recall = len(gold_ids & predicted) / len(gold_ids) if gold_ids else 0.0
        slices = [name for name, pattern in PATTERNS.items() if re.search(pattern, text)]
        slices.append("multi_gold" if len(gold_ids) > 1 else "single_gold")
        slices.append("all")
        for name in slices:
            stats[name][0] += 1
            stats[name][1] += recall
            stats[name][2] += int(recall < 1.0)

    for name, (count, recall_sum, imperfect) in sorted(
        stats.items(), key=lambda item: item[1][1] / item[1][0]
    ):
        print(f"{name:18s} n={count:4d} recall={recall_sum / count:.4f} imperfect={imperfect:4d}")


if __name__ == "__main__":
    main()
