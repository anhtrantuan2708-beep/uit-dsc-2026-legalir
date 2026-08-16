# LegalIR V12 document-disjoint validation

Date: 2026-08-16

## Why the old validation failed

The original query split shared 481 of 776 unique Dev gold document IDs with
labelled Train (62%). Query-fold CV therefore allowed models to learn document
patterns that did not transfer reliably to Public.

V12 creates connected components of queries that share any gold document, then
assigns whole components to Train or Dev. The new split has 1,008 Dev queries
and zero gold-document overlap with its 5,992-query Train split.

## Baselines on unseen documents

| Retriever | Recall@5 | Recall@100 |
|---|---:|---:|
| Character question-KNN | 0.0000 | 0.0000 |
| Dense question-KNN | 0.0000 | 0.0000 |
| BM25 corpus retrieval | 0.3221 | 0.6849 |
| Dense corpus retrieval | 0.2913 | 0.7455 |
| BM25 + dense RRF | **0.4320** | **0.8457** |

Question-KNN can only return document IDs observed in labelled Train, so zero
Recall is expected on a fully document-disjoint split. Corpus retrieval is the
required fallback for unseen legal documents.

## BGE reranking smoke validation

Two non-overlapping deterministic 100-query blocks were tested by reranking the
first 20 BM25+dense candidates with BGE:

| Block | Hybrid Recall@5 | BGE Recall@5 | Gain |
|---|---:|---:|---:|
| A | 0.4750 | 0.6250 | +0.1500 |
| B | 0.4008 | 0.5892 | +0.1884 |

The gain repeated on both blocks. Full document-disjoint validation is running
in resumable 200-query shards via
`scripts/resume_legalir_docdisjoint_bge.sh`. This run does not create or upload
a Public submission.
