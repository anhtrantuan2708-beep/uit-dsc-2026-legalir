# Listwise reranker — 300-query candidate V7

Date: 2026-08-13

## Hypothesis

The earlier pointwise hard-negative rerankers learned each question-document
pair independently and underperformed V6. A listwise objective should better
match the competition decision: rank several candidate legal documents for the
same question and select the best five.

## Setup

- 300 labelled train queries, separate from the 1,003-query dev split.
- All available positive documents plus four RRF-mined hard negatives/query.
- LambdaLoss optimized at `k=5`, one epoch, batch size 2, MPS.
- Two evidence chunks per document, maximum sequence length 384.
- Training time: 126 seconds.
- Full-dev inference: 50,150 query-document pairs, about 12 minutes including
  evidence preparation.

## Results

| Candidate | Recall | Precision | Hit@5 |
|---|---:|---:|---:|
| V6 generic reranker, keep 3 | 0.8264 | 0.1753 | 0.8524 |
| Listwise, keep 1 | 0.8045 | 0.1707 | 0.8295 |
| Listwise, keep 2 | 0.8188 | 0.1737 | 0.8435 |
| Listwise, keep 3 | 0.8313 | 0.1765 | 0.8564 |
| **Listwise, keep 4** | **0.8339** | **0.1769** | **0.8594** |

The best listwise blend improves full-dev Recall by 0.0075 and Precision by
0.0016 over V6. It retrieves more relevant documents on 26 queries, fewer on
18, and has a net gain of eight relevant documents.

## Decision

Accept as **V7 local candidate**, but do not replace the Public best until its
submission archive is generated, validated, and scored. The gain is positive
on full dev but small enough that a cross-validation check is still useful
before spending another Public submission.
