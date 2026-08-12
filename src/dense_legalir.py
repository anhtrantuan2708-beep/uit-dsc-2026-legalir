#!/usr/bin/env python3
"""Local multilingual dense retrieval baseline for LegalIR.

Uses E5 query/passage prefixes and keeps model + embeddings on the local machine.
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> list[dict]:
    if path.is_file():
        rows = load_json(path)
        return [row for row in rows if isinstance(row, dict) and "id" in row and "passage" in row]
    records = []
    for file in sorted(path.glob("*.json")):
        row = load_json(file)
        if isinstance(row, dict) and "id" in row and "passage" in row:
            records.append(row)
    return records


def document_text(row: dict, max_chars: int) -> str:
    title = str(row.get("name", "")).replace("-", " ")
    passage = str(row["passage"])[:max_chars]
    return f"passage: {title}\n{passage}"


def load_or_encode_corpus(model, corpus: list[dict], embedding_path: Path, ids_path: Path, batch_size: int, max_chars: int):
    import numpy as np

    doc_ids = [str(row.get("source_id", row["id"])) for row in corpus]
    if embedding_path.exists() and ids_path.exists():
        cached_ids = load_json(ids_path)
        if cached_ids == doc_ids:
            print(f"reusing local corpus embeddings: {embedding_path}")
            return np.load(embedding_path), doc_ids

    texts = [document_text(row, max_chars) for row in corpus]
    print(f"encoding {len(texts)} corpus passages locally")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings)
    ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
    return embeddings, doc_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--model-cache", type=Path, default=Path("models/e5-small"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/derived/e5_corpus_embeddings.npy"))
    parser.add_argument("--embedding-ids", type=Path, default=Path("data/derived/e5_corpus_ids.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Dense dependencies missing. Run: uv pip install --python .venv/bin/python -r requirements-dense.txt"
        ) from error

    queries = load_json(args.queries)
    corpus = load_corpus(args.corpus)
    if not corpus:
        raise SystemExit("No corpus records found")

    model = SentenceTransformer(args.model, cache_folder=str(args.model_cache))
    corpus_embeddings, doc_ids = load_or_encode_corpus(
        model, corpus, args.embeddings, args.embedding_ids, args.batch_size, args.max_chars
    )

    query_ids = list(queries)
    query_texts = [f"query: {queries[query_id]['question']}" for query_id in query_ids]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    scores = query_embeddings @ corpus_embeddings.T
    # Keep a longer internal ranking for fusion/reranking.  Submission files are
    # checked separately and must contain no more than five document IDs.
    k = min(args.top_k, len(doc_ids))
    result = {}
    for index, query_id in enumerate(query_ids):
        candidate_count = min(len(doc_ids), max(k * 20, k))
        candidate_indices = np.argpartition(scores[index], -candidate_count)[-candidate_count:]
        ranked = candidate_indices[np.argsort(scores[index][candidate_indices])[::-1]]
        answer = []
        seen = set()
        for item_index in ranked:
            document_id = doc_ids[int(item_index)]
            if document_id not in seen:
                answer.append(document_id)
                seen.add(document_id)
            if len(answer) == k:
                break
        result[query_id] = {"answer": answer}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} dense predictions to {args.output}")


if __name__ == "__main__":
    main()
