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

## Decision

Accept V10 as the current local candidate. Public inference still needs the
same retrieval branches, V9 fusion and BGE top-10 reranking before a validated
submission archive can be created.
