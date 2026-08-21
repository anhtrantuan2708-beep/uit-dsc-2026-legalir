#!/usr/bin/env python3
"""Locally rerank LegalIR candidates with a multilingual cross-encoder.

The retrievers find broadly relevant documents.  This script lets a second
model read a question together with each candidate and order the shortlist.
Only local files are passed to the model after its public weights are cached.
"""

import argparse
import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


def split_chunks(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    return [text[start : start + size] for start in range(0, len(text), size - overlap)]


def best_evidence(query: str, document: dict, count: int) -> str:
    """Keep the most query-related portions of a long statute."""
    query_tokens = token_set(query)
    chunks = split_chunks(str(document["passage"]))
    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (-len(query_tokens & token_set(item[1])), item[0]),
    )
    selected = [chunk for _, chunk in ranked[:count]]
    title = str(document.get("name", "")).replace("-", " ")
    return f"{title}\n" + "\n...\n".join(selected)


def load_corpus(path: Path, needed_ids: set[str]) -> dict[str, dict]:
    """Read only documents which actually occur in the candidate shortlist."""
    records = {}
    for document_id in needed_ids:
        file = path / f"context_{document_id}.json"
        if not file.exists():
            continue
        row = load_json(file)
        if isinstance(row, dict) and "id" in row and "passage" in row:
            records[str(row["id"])] = row
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--model-cache", type=Path, default=Path("models/mmarco-minilm"))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-chunks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scores-output", type=Path,
                        help="Optional confidence scores for safe routing experiments")
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Temporary smoke-test limit. Omit for a full validation run.",
    )
    parser.add_argument(
        "--query-start",
        type=int,
        default=0,
        help="Zero-based query offset, useful for resumable inference shards.",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise SystemExit("Install requirements-dense.txt first.") from error

    queries = load_json(args.queries)
    candidate_rows = load_json(args.candidates)
    query_items = list(queries.items())[args.query_start :]
    if args.max_queries is not None:
        query_items = query_items[: args.max_queries]
    needed_ids = {
        str(document_id)
        for query_id, _ in query_items
        for document_id in candidate_rows.get(query_id, {}).get("answer", [])[: args.candidate_k]
    }
    corpus = load_corpus(args.corpus, needed_ids)
    print(f"loaded {len(corpus)} unique candidate documents")
    pairs, locations = [], []
    for query_id, row in query_items:
        question = str(row["question"])
        for rank, document_id in enumerate(candidate_rows.get(query_id, {}).get("answer", [])[: args.candidate_k]):
            document_id = str(document_id)
            document = corpus.get(document_id)
            if document is None:
                continue
            pairs.append([question, best_evidence(question, document, args.evidence_chunks)])
            locations.append((str(query_id), rank, document_id))

    print(f"scoring {len(pairs)} question-document pairs locally")
    model = CrossEncoder(args.model, cache_folder=str(args.model_cache))
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)

    by_query: dict[str, list[tuple[float, int, str]]] = {str(query_id): [] for query_id, _ in query_items}
    for score, (query_id, rank, document_id) in zip(scores, locations):
        by_query[query_id].append((float(score), rank, document_id))

    result = {}
    score_rows = {}
    for query_id, ranked in by_query.items():
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        result[query_id] = {"answer": [document_id for _, _, document_id in ranked[: args.top_k]]}
        score_rows[query_id] = [
            {"id": document_id, "score": score, "candidate_rank": rank + 1}
            for score, rank, document_id in ranked
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(result)} reranked predictions to {args.output}")
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        args.scores_output.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote reranker scores to {args.scores_output}")


if __name__ == "__main__":
    main()
