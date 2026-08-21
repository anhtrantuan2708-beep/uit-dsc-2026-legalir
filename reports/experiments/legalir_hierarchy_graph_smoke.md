# Hierarchy FTS and reference-graph smoke — 21/08/2026

## Scope

This note records controlled local experiments after V12.  None is a Public
Codabench result and neither replaces the preserved V10 submission.

## T01 — hierarchy-first FTS candidate retrieval

- Corpus parsed: 8,532 documents; 1,673,053 child units and 257,354 parent
  evidence units.  `Điều` headings were found in 7,182 documents; remaining
  documents use fallback units, so no document was dropped.
- On three fixed smoke blocks, flat FTS candidate Recall@100 was `0.9239`;
  hierarchy MaxP was `0.9447` (`+0.0208`).  Per-block: A `0.9542`, B `0.9300`,
  C `0.9500`.
- Keep as a candidate source.  A direct BGE rerank on the first ten queries
  did not beat V12, so do not scale that reranker variant.

## T03 — one-hop legal reference graph

- Conservative extraction created 21,548 edges from 8,532 organizer documents
  (5,361 source nodes).  52,394 of 129,113 detected references resolved.
- Oracle expansion from V12 top-20 increased candidate Recall `0.8560` to
  `0.8816` (`+0.0256`); multi-answer queries increased `+0.0813`.
- Actual safe top-5 expansion failed.  The best tested setting (keep four V12
  IDs; use graph neighbours from the first three seeds) scored local Recall
  `0.8307`, below V12 `0.8671`.

## Decision

Do **not** use graph neighbours in a submission.  Retain the graph artifact as
candidate-analysis data only.  Next gated experiment: BGE-M3 hybrid candidate
retrieval, first on the same fixed smoke blocks.

## Follow-up: semantic and representation gates

- `BAAI/bge-m3` weights were cached, but full-corpus encoding did not start
  safely on this 16 GB MPS machine.  It was stopped before creating a ranking.
- `dangvantuan/vietnamese-document-embedding` was incompatible with the
  installed Transformers runtime on both MPS and CPU; it was rejected at a
  20-document preflight.
- `bkai-foundation-models/vietnamese-bi-encoder` encoded successfully, but
  dense Recall@100 on smoke A was `0.5467`.  RRF with FTS was `0.9217`; even
  very small dense weights only tied flat FTS `0.9467`.  Stop: no B/C or full
  Dev run.Í
- Fusing flat FTS + hierarchy MaxP is the surviving candidate change:
  candidate Recall@100 A/B/C = `0.9542 / 0.9300 / 0.9650`, pooled `0.9497`.
  This is above flat FTS pooled `0.9239` and hierarchy-alone `0.9447`.
- A 20-query safe rerank of its top-20 candidates with BGE v2-m3 made no
  change to V12 Recall@5 on smoke A (`0.8817`); do not scale.
- Adding metadata-derived document titles to hierarchy paths also tied the
  original hierarchy score on smoke A (`0.9542@100`), so it is not a separate
  retrieval branch.

## Current bottleneck

Candidate recall is no longer the primary blocker.  The next experiment must
improve the *candidate-to-top-5* decision with stronger evidence supervision;
the V14 weak lexical-chunk fine-tune remains rejected.

## Final smoke: multi-evidence reranker

- `rerank_fused_fts_evidence_legalir.py` was added to score, for each fused
  candidate document, both its flat-FTS chunk and hierarchy-FTS child then take
  the higher cross-encoder score.
- On the first 20 queries of smoke A, V12 routing with four protected base IDs
  stayed at `0.8817@5` for thresholds 0.70–0.96.  The cross-encoder scores were
  saturated for many incorrect candidates, so this is rejected before a
  100-query or full-Dev run.
