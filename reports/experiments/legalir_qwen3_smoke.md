# Qwen3 reranker smoke — 21/08/2026

## Scope

This is a local, gated test of `Qwen/Qwen3-Reranker-0.6B`. It did not create a
Codabench submission and does not replace the preserved V10 Public candidate.

## Setup

- Device: Apple M1 Pro / 16 GB, MPS.
- Inference only; no fine-tuning, no corpus embedding, and no external API.
- For each query, the candidate pool was the ordered union of V12 top-5, flat
  FTS top-20, and hierarchy-FTS top-20. Duplicate document IDs were removed.
- Qwen scored at most two matched chunks per document: one flat FTS chunk and
  one hierarchy child. `max_length=512`, `batch_size=1`.
- Prompt: `Given a Vietnamese legal question, judge whether the document
  contains legal provisions necessary to answer it.`

## Runtime smoke (10 fixed queries)

- Qwen loaded and ran on MPS without an out-of-memory error.
- It scored 50--56 evidence pairs/query in about 10--12 seconds/query.
- The V12 baseline was already Recall `1.0000` / Precision `0.2000` on this
  small sample. Qwen direct dropped to Recall `0.9000` / Precision `0.1800`.
- Keeping V12's first four IDs and letting Qwen fill the fifth made no change.
  This sample was therefore only a runtime pass, not a quality pass.

## Block A (100 fixed queries)

| Method | Recall@5 | Precision@5 |
|---|---:|---:|
| V12 base | **0.8817** | **0.1900** |
| Qwen direct top-5 | 0.8092 | 0.1740 |
| Best safe router | 0.8817 | 0.1900 |

The union candidate pool had Recall `0.8817@5`, `0.9067@20`, and `0.9217@45`.
Thus Qwen had recoverable candidates available but ranked them less accurately
than V12.

The safe-router calibration tested 30 score/margin settings. No setting
improved V12. The best selected rule was the no-op: route zero queries. The
best non-no-op settings also tied V12, while a broad rule routing 90 queries
fell to Recall `0.8592` / Precision `0.1840`.

Runtime for 100 queries was `1,449.7` seconds (`12.0` seconds/query), with
process peak RSS `2,063.4 MB`. The macOS swap figure was a machine-wide snapshot
and must not be attributed entirely to Qwen.

## Decision: REJECT before Block B

Block B cannot meet the required `+0.01` Recall gate because the only A-selected
configuration routes zero queries. Running another 100 Qwen queries would only
reproduce V12 and consume roughly 20 minutes on this Mac. Do not run full Dev
or make a Public submission from this branch.

## Reproducibility artifacts

- Runner: `src/rerank_qwen3_legalir.py`
- Router calibration: `src/calibrate_legalir_router.py`
- A predictions/scores/candidate coverage: `tmp/qwen3_a100_*`

Next bounded challenger: `jinaai/jina-reranker-v3.5-mlx` on the same 10-query
runtime gate and the same A/B protocol. It must not be downloaded or scaled
until explicitly started.
