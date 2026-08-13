#!/usr/bin/env python3
"""Fine-tune a LegalIR cross-encoder with a listwise ranking objective."""

import argparse
import json
import random
from pathlib import Path

from finetune_legalir_crossencoder import load_needed_corpus, mine_candidates
from rerank_legalir import best_evidence


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("rankings", type=Path, nargs="+")
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--model-cache", type=Path, default=Path("models/mmarco-minilm"))
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--evidence-chunks", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import LambdaLoss

    random.seed(args.seed)
    train = load(args.labelled_train)
    rankings = [load(path) for path in args.rankings]
    query_ids = list(train)
    random.shuffle(query_ids)
    query_ids = query_ids[: args.max_queries]

    selections = []
    needed = set()
    for query_id in query_ids:
        positives = [str(value) for value in train[query_id].get("answer", [])]
        negatives = [
            document_id
            for document_id in mine_candidates(query_id, rankings, args.candidate_k)
            if document_id not in positives
        ][: args.negatives]
        if not positives or len(negatives) < args.negatives:
            continue
        selections.append((query_id, positives, negatives))
        needed.update(positives)
        needed.update(negatives)

    corpus = load_needed_corpus(args.corpus, needed)
    rows = {"query": [], "docs": [], "labels": []}
    for query_id, positives, negatives in selections:
        question = str(train[query_id]["question"])
        positives = [document_id for document_id in positives if document_id in corpus]
        negatives = [document_id for document_id in negatives if document_id in corpus]
        if not positives or len(negatives) < args.negatives:
            continue
        documents = positives + negatives
        rows["query"].append(question)
        rows["docs"].append(
            [best_evidence(question, corpus[document_id], args.evidence_chunks) for document_id in documents]
        )
        rows["labels"].append([1.0] * len(positives) + [0.0] * len(negatives))

    if not rows["query"]:
        raise RuntimeError("No listwise training rows were created.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(
        args.model,
        cache_folder=str(args.model_cache),
        device=device,
        num_labels=1,
        max_length=args.max_length,
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.output_model),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        dataloader_pin_memory=False,
    )
    dataset = Dataset.from_dict(rows)
    print(
        f"listwise rows: {len(dataset)} | documents: {sum(map(len, rows['docs']))} | "
        f"device: {device} | max_length: {args.max_length}",
        flush=True,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=LambdaLoss(model, k=5, mini_batch_size=8),
    )
    trainer.train()
    model.save(str(args.output_model))
    print(f"saved listwise cross-encoder to {args.output_model}", flush=True)


if __name__ == "__main__":
    main()
