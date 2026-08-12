#!/usr/bin/env python3
"""Fine-tune the LegalIR cross-encoder with mined hard negatives."""

import argparse
import json
import random
from pathlib import Path

from rerank_legalir import best_evidence


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_needed_corpus(path: Path, needed: set[str]) -> dict[str, dict]:
    result = {}
    for document_id in needed:
        file = path / f"context_{document_id}.json"
        if file.exists():
            row = load(file)
            if isinstance(row, dict) and "passage" in row:
                result[document_id] = row
    return result


def mine_candidates(query_id: str, rankings: list[dict], limit: int) -> list[str]:
    score = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking.get(query_id, {}).get("answer", [])[:limit], start=1):
            document_id = str(document_id)
            score[document_id] = score.get(document_id, 0.0) + 1 / (15 + rank)
    return sorted(score, key=lambda document_id: (-score[document_id], document_id))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labelled_train", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("rankings", type=Path, nargs="+")
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--model-cache", type=Path, default=Path("models/mmarco-minilm"))
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--evidence-chunks", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder, InputExample

    random.seed(args.seed)
    train = load(args.labelled_train)
    rankings = [load(path) for path in args.rankings]
    query_ids = list(train)
    random.shuffle(query_ids)
    query_ids = query_ids[: args.max_queries]

    selections = []
    needed = set()
    for query_id in query_ids:
        gold = [str(value) for value in train[query_id].get("answer", [])]
        negatives = [
            document_id
            for document_id in mine_candidates(query_id, rankings, args.candidate_k)
            if document_id not in gold
        ][: args.negatives]
        if not gold or len(negatives) < args.negatives:
            continue
        positive = gold[0]
        selections.append((query_id, positive, negatives))
        needed.add(positive)
        needed.update(negatives)

    corpus = load_needed_corpus(args.corpus, needed)
    examples = []
    for query_id, positive, negatives in selections:
        question = str(train[query_id]["question"])
        if positive not in corpus:
            continue
        examples.append(
            InputExample(texts=[question, best_evidence(question, corpus[positive], args.evidence_chunks)], label=1.0)
        )
        for negative in negatives:
            if negative in corpus:
                examples.append(
                    InputExample(texts=[question, best_evidence(question, corpus[negative], args.evidence_chunks)], label=0.0)
                )

    random.shuffle(examples)
    if not examples:
        raise RuntimeError("No training examples were created; check labels, rankings, and corpus paths.")
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"training examples: {len(examples)} | queries: {len(selections)} | "
        f"device: {device} | max_length: {args.max_length}",
        flush=True,
    )
    model = CrossEncoder(
        args.model,
        cache_folder=str(args.model_cache),
        device=device,
        num_labels=1,
        max_length=args.max_length,
    )
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    print(f"training steps: {len(loader) * args.epochs}", flush=True)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=max(1, int(len(loader) * args.epochs * 0.1)),
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_model),
        show_progress_bar=not args.no_progress,
    )
    # sentence-transformers 5.x does not persist output_path at train end when
    # fit() has no evaluator, so save explicitly instead of trusting the legacy
    # output_path behaviour.
    model.save(str(args.output_model))
    if not (args.output_model / "config.json").exists():
        raise RuntimeError(f"Training ended but the model was not saved to {args.output_model}")
    print(f"saved hard-negative cross-encoder to {args.output_model}", flush=True)


if __name__ == "__main__":
    main()
