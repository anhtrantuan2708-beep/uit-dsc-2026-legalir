# Listwise reranker — 800-query scale check

Date: 2026-08-13

- Training: 800 queries, 4,087 documents, LambdaLoss at top 5, one epoch.
- Runtime: 646 seconds on MPS.
- Fixed smoke100, best keep-base=4: Recall 0.8558, Precision 0.1820,
  Hit@5 0.9000.
- Existing V7-300 on the same split: Recall 0.8558, Precision 0.1820,
  Hit@5 0.9000.

Decision: reject before full-dev inference. Scaling from 300 to 800 queries did
not improve the fixed smoke gate, so V7-300 remains the local candidate.
