# LegalIR V11 learned fusion

Date: 2026-08-16

## Hypothesis

The candidate pool already has high coverage, but fixed RRF and BGE do not learn
which retriever is trustworthy for each document. A small supervised fusion
model can learn this from ranking positions without rerunning embeddings.

## Method

- Nine cached lexical, dense, question-neighbour, listwise and BGE rankings.
- Candidate features contain presence and rank evidence from every source plus
  consensus and document-frequency features.
- HistGradientBoosting with seven leaves, 150 iterations and balanced classes.
- Five-fold GroupKFold by query; every reported dev prediction is out-of-fold.

## Result

| Candidate | Recall | Precision | Decision |
|---|---:|---:|---|
| V10 local | 0.8562 | 0.1821 | Baseline |
| V11, 15 leaves | 0.8675 | 0.1846 | Improved, below target |
| V11, 31 leaves | 0.8628 | 0.1838 | Reject |
| **V11, 7 leaves** | **0.8728** | **0.1858** | **Accept** |

V11 improves Recall by 0.0166 over V10 local, so it passes the required +0.01
gate. Public inference reuses cached rankings and takes only seconds.

## Artifact

`submissions/legalir_public_v11.zip`

SHA-256:
`3804d1b854ae7456d9d3efc1e9c1f37ef5c54995e0fa7c5d640a11e254f63ebd`
