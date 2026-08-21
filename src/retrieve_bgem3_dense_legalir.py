#!/usr/bin/env python3
"""Dense LegalIR retrieval with reproducible on-disk caches.

This script deliberately does one thing: produce a top-k document ranking from
the named SentenceTransformer model.  Sparse retrieval remains the tested FTS5
ranker and can be combined with this output using ``fuse_legalir_multi.py``.
Keeping the two sources separate makes a smoke-test failure easy to diagnose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def document_text(row: dict, max_chars: int) -> str:
    title = " ".join(str(row.get("name", "")).replace("-", " ").split())
    body = " ".join(str(row.get("passage", "")).split())[:max_chars]
    return f"{title}\n{body}".strip()


def fingerprint(rows: list[dict], model_name: str, max_chars: int, max_seq_length: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model_name}|{max_chars}|{max_seq_length}".encode())
    for row in rows:
        digest.update(str(row["id"]).encode())
        digest.update(document_text(row, max_chars).encode())
    return digest.hexdigest()


def encode_documents(model, texts: list[str], batch_size: int):
    if hasattr(model, "encode_document"):
        return model.encode_document(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)


def encode_queries(model, texts: list[str], batch_size: int):
    if hasattr(model, "encode_query"):
        return model.encode_query(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("corpus", type=Path, help="Representative corpus JSON list")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/cache/bge-m3"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Required only by model cards that provide custom embedding code")
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    queries = load(args.queries)
    corpus = load(args.corpus)
    if not isinstance(corpus, list) or not corpus:
        raise SystemExit("Corpus must be a non-empty JSON list")
    if not all(isinstance(row, dict) and "id" in row for row in corpus):
        raise SystemExit("Every corpus row needs an id")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_value = fingerprint(corpus, args.model, args.max_chars, args.max_seq_length)
    matrix_path = args.cache_dir / f"corpus_{fingerprint_value[:16]}.npy"
    metadata_path = args.cache_dir / f"corpus_{fingerprint_value[:16]}.json"
    ids = [str(row["id"]) for row in corpus]

    model = None
    if matrix_path.exists() and metadata_path.exists():
        metadata = load(metadata_path)
        if metadata.get("ids") == ids and metadata.get("fingerprint") == fingerprint_value:
            corpus_embeddings = np.load(matrix_path)
            print(f"reusing dense corpus cache: {matrix_path}")
        else:
            matrix_path.unlink(missing_ok=True)
            model = None
    if not matrix_path.exists():
        model = SentenceTransformer(
            args.model, cache_folder=str(args.cache_dir / "hf"), device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        model.max_seq_length = args.max_seq_length
        corpus_embeddings = encode_documents(
            model, [document_text(row, args.max_chars) for row in corpus], args.batch_size
        )
        np.save(matrix_path, corpus_embeddings)
        metadata_path.write_text(json.dumps({"fingerprint": fingerprint_value, "ids": ids}, ensure_ascii=False), encoding="utf-8")
        print(f"saved dense corpus cache: {matrix_path}")
    if model is None:
        model = SentenceTransformer(
            args.model, cache_folder=str(args.cache_dir / "hf"), device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        model.max_seq_length = args.max_seq_length

    query_ids = list(queries)
    query_embeddings = encode_queries(
        model, [str(queries[query_id]["question"]) for query_id in query_ids], args.batch_size
    )
    scores = query_embeddings @ corpus_embeddings.T
    internal_k = min(len(ids), max(args.top_k * 2, args.top_k))
    result = {}
    for index, query_id in enumerate(query_ids):
        indices = np.argpartition(scores[index], -internal_k)[-internal_k:]
        indices = indices[np.argsort(scores[index][indices])[::-1]]
        result[str(query_id)] = {"answer": [ids[int(item)] for item in indices[: args.top_k]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} dense rankings to {args.output}")


if __name__ == "__main__":
    main()
