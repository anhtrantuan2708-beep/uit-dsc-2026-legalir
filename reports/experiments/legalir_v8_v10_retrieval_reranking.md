# LegalIR V8–V10 retrieval and reranking experiments

Date: 2026-08-13

## V8 — character-ngram candidate retrieval

- Added character 3–5 gram TF-IDF nearest-question retrieval.
- Candidate Recall@100 increased from 0.9422 to 0.9534 after weighted fusion.
- Listwise reranking, keep-base=3: Recall 0.8367, Precision 0.1777,
  Hit@5 0.8614.

## V9 — tuned dense-question voting

- Tuned supervised dense question-KNN to 50 neighbours, similarity power 4,
  and rank power 0.5.
- Fused its vote with V8 using RRF weight 0.5 and `rrf-k=1`.
- Full dev: Recall 0.8504, Precision 0.1815, Hit@5 0.8754.
- Three deterministic query-ID folds: 0.8706, 0.8489, 0.8329.

## V10 — BGE multilingual reranker

- Used `BAAI/bge-reranker-v2-m3` only on V9's first ten candidates.
- Direct BGE ranking with the first three trusted base IDs retained was best.
- Full dev: Recall **0.8562**, Precision **0.1821**, Hit@5 **0.8814**.
- V9 full dev: Recall 0.8504, Precision 0.1815, Hit@5 0.8754.

## Rejected checks

- Listwise seed ensemble: tied V7, no gain.
- Word TF-IDF and character/word ensemble: increased candidate recall but did
  not improve final top 5 over V8.
- BGE over the original top 20 was weaker alone; reranking V9's top ten was
  more efficient and produced the best full-dev result.
- Document-association fusion improved V9 only from 0.8504 to 0.8534 Recall.
- A character-TF-IDF classifier that applied association expansion only to
  likely multi-document queries reached 0.8524 Recall. Both remain below the
  +0.01 experiment gate and were not scaled to Public.

V9 error slicing shows the main bottleneck is ranking, especially for queries
with multiple gold documents: top-5 Recall is 0.6087 for that slice, while the
same candidate top 50 contains 0.8404. Overall candidate Recall@50 is 0.9381,
compared with final top-5 Recall 0.8504.

## Decision

V9 Public scored Recall **0.84625** and Precision **0.1812**, close to its local
Recall 0.8504 and Precision 0.1815. Accept V9 as the current Public baseline.

V10 Public inference has completed and produced the validated candidate
`submissions/legalir_public_v10.zip`. Its Codabench score is still unknown.

The full reproducible Public pipeline is `scripts/run_legalir_public_v10.sh`.
The BGE-only resumable stage is `scripts/resume_legalir_public_v10_bge.sh`.
Public BGE inference is split into ten 100-query shards so interruptions reuse
completed work instead of restarting all 1,000 queries.
