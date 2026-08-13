# Hard-negative reranker v2 — 800-query smoke

Date: 2026-08-13

## Hypothesis

Training the multilingual cross-encoder on more LegalIR queries, all available
gold documents, and four hard negatives per query may improve final top-5
selection beyond the pretrained generic V6 reranker.

## Setup

- 800 labelled train queries, held separate from the 1,003-query dev split.
- 4,087 pointwise training pairs; 1 epoch; 511 steps; MPS.
- All gold documents per query, four RRF-mined hard negatives.
- Two evidence chunks, maximum sequence length 384.
- Training time: 364 seconds on Apple M1 Pro 16 GB.

## Fixed 100-query smoke result

| Candidate | Recall | Precision | Hit@5 |
|---|---:|---:|---:|
| V6 generic reranker | **0.8508** | **0.1800** | **0.8900** |
| Hard-negative v2, best blend | 0.8058 | 0.1720 | 0.8500 |

The trained model improves only one query and hurts five versus V6 on this
slice. RRF ensembling with V6 also decreases Recall.

## Decision

Reject and do not run full-dev/Public inference. The next experiment should
change the training objective or add legal-aware features, rather than merely
scaling this pointwise binary-loss recipe.
