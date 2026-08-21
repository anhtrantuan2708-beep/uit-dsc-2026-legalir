#!/usr/bin/env python3
"""Fine-tune a cross-encoder on relevant legal chunks and FTS hard negatives."""

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from chunk_legalir_contexts import split_document
from fts_chunk_legalir import query_expression


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lexical_overlap(question: str, passage: str) -> int:
    question_words = Counter(query_expression(question).replace('"', '').replace(' OR ', ' ').split())
    passage_words = Counter(query_expression(passage).replace('"', '').replace(' OR ', ' ').split())
    return sum(min(count, passage_words[word]) for word, count in question_words.items())


def best_gold_chunk(question: str, corpus: Path, document_id: str, chunk_size: int, overlap: int) -> tuple[str, str] | None:
    path = corpus / f"context_{document_id}.json"
    if not path.exists():
        return None
    row = load(path)
    chunks = split_document(str(row.get("passage", "")), chunk_size, overlap)
    if not chunks:
        return None
    best = max(chunks, key=lambda chunk: lexical_overlap(question, chunk))
    return str(row.get("name", "")), best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--model-cache", type=Path, default=Path("models/mmarco-minilm"))
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--chunk-k", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    train = load(args.labelled_train)
    query_ids = list(train)
    random.Random(args.seed).shuffle(query_ids)
    query_ids = query_ids[: args.max_queries]
    connection = sqlite3.connect(args.database)
    examples = []
    usable = 0
    for query_id in query_ids:
        question = str(train[query_id]["question"])
        gold_ids = [str(value) for value in train[query_id].get("answer", [])]
        if not gold_ids:
            continue
        positive = best_gold_chunk(question, args.corpus, gold_ids[0], args.chunk_size, args.overlap)
        expression = query_expression(question)
        if positive is None or not expression:
            continue
        negatives = []
        rows = connection.execute(
            "SELECT source_id, name, passage FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks, 0.0, 2.0, 1.0) LIMIT ?",
            (expression, args.chunk_k),
        )
        seen = set()
        for source_id, name, passage in rows:
            source_id = str(source_id)
            if source_id in seen:
                continue
            seen.add(source_id)
            if source_id not in gold_ids:
                negatives.append((str(name), str(passage)))
            if len(negatives) == args.negatives:
                break
        if len(negatives) < args.negatives:
            continue
        usable += 1
        positive_name, positive_passage = positive
        examples.append(InputExample(texts=[question, f"{positive_name}\n{positive_passage}"], label=1.0))
        examples.extend(InputExample(texts=[question, f"{name}\n{passage}"], label=0.0) for name, passage in negatives)
    connection.close()
    if not examples:
        raise RuntimeError("No chunk training examples were created")

    random.Random(args.seed).shuffle(examples)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"training examples: {len(examples)} | usable queries: {usable} | device: {device}", flush=True)
    model = CrossEncoder(args.model, cache_folder=str(args.model_cache), device=device, num_labels=1, max_length=args.max_length)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=max(1, int(len(loader) * args.epochs * 0.1)),
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_model),
        show_progress_bar=not args.no_progress,
    )
    model.save(str(args.output_model))
    print(f"saved chunk reranker to {args.output_model}", flush=True)


if __name__ == "__main__":
    main()
