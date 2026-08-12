#!/usr/bin/env python3
"""Fine-tune local multilingual E5 on labelled LegalIR question-document pairs.

Use a small smoke run first.  Each batch's other documents act as in-batch hard
negatives, teaching the model to prefer the matching statute over related laws.
"""

import argparse
import json
import random
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_corpus(path: Path, needed: set[str]) -> dict[str, dict]:
    result = {}
    for document_id in needed:
        file = path / f"context_{document_id}.json"
        if file.exists():
            row = load(file)
            if isinstance(row, dict) and "passage" in row:
                result[document_id] = row
    return result


def document_text(row: dict, max_chars: int) -> str:
    title = str(row.get("name", "")).replace("-", " ")
    passage = " ".join(str(row["passage"]).split())[:max_chars]
    return f"passage: {title}\n{passage}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--model-cache", type=Path, default=Path("models/e5-small"))
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample, SentenceTransformer, losses

    random.seed(args.seed)
    train = load(args.labelled_train)
    raw_pairs = [
        (str(row["question"]), str(document_id))
        for row in train.values()
        for document_id in row.get("answer", [])
    ]
    random.shuffle(raw_pairs)
    raw_pairs = raw_pairs[: args.max_pairs]
    corpus = read_corpus(args.corpus, {document_id for _, document_id in raw_pairs})
    examples = [
        InputExample(texts=[f"query: {question}", document_text(corpus[document_id], args.max_chars)])
        for question, document_id in raw_pairs
        if document_id in corpus
    ]
    if len(examples) < args.batch_size:
        raise SystemExit("Not enough valid training pairs")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"training pairs: {len(examples)} | device: {device} | epochs: {args.epochs}")
    model = SentenceTransformer(args.model, cache_folder=str(args.model_cache), device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(loader) * args.epochs * 0.1))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_model),
        show_progress_bar=True,
    )
    print(f"saved fine-tuned model to {args.output_model}")


if __name__ == "__main__":
    main()
