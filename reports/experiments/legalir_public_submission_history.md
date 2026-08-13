# LegalIR Public Test — Submission history

| Version | Codabench submission | Public Recall | Public Precision | Decision |
|---|---:|---:|---:|---|
| V1: BM25 + Dense RRF | 886134 | 0.5843 | 0.1248 | Baseline |
| V3: supervised 3-way hybrid | 886603 | 0.7502 | 0.1610 | Keep |
| V5: V4 + safe rerank fifth slot | 886623 | **0.8020** | **0.1724** | **Current best** |

V5 retains the first four IDs from the V4 multi-retriever result, and uses the
cross-encoder reranker only to fill the fifth slot.  Its local dev result was
Recall 0.8059 and Precision 0.1703, which closely matches its Public Recall.

## Rejected experiment: E5 fine-tune smoke (2026-08-12)

- Train setup: 100 labelled question-document pairs, 1 epoch, local MPS.
- Direct dense retrieval on the fixed 100-query dev smoke set: Recall 0.5050.
- Replacing the original dense branch in V4 fusion: Recall 0.7567.
- Baseline V4 on the same smoke set: Recall 0.7833.

Decision: do not scale this naïve fine-tune. It lacks mined hard negatives and
is worse than the existing pretrained E5 branch.

## Rejected experiment: hard-negative cross-encoder smoke (2026-08-13)

- Train setup: 20 labelled queries, 1 positive and 2 mined negatives per query,
  8 MPS training steps, maximum sequence length 256.
- Direct reranking on the fixed 100-query dev smoke set: Recall 0.7233.
- Safe blend keeping four V4 results: Recall 0.8133.
- Existing V5 on the same smoke set: Recall 0.8133.

Decision: do not scale this configuration because it adds no local Recall over
V5.  The training script now explicitly calls `model.save()` because the
installed sentence-transformers 5.x `fit(output_path=...)` path does not save
at train end when no evaluator is supplied.

## Candidate V6: deep generic reranking (2026-08-13)

- Candidate ceiling analysis on 1,003 dev queries: Recall@10 0.8298,
  Recall@20 0.8764, Recall@50 0.9177, and Recall@100 0.9422.
- Rerank the first 50 candidates with the pretrained multilingual mMARCO
  cross-encoder, keep the first 3 trusted fusion results, and fill slots 4-5
  from the reranker.
- Fixed 100-query smoke: Recall 0.8558 when keeping four base results.
- Full dev, tuned only across keep-base values 1-4: Recall **0.8264**,
  Precision **0.1753**, Hit@5 **0.8524** with keep-base=3.
- Existing V5 full dev: Recall 0.8059, Precision 0.1703, Hit@5 0.8325.

Decision: the +0.0205 full-dev Recall gain passes the local gate. Generate a
Public candidate, validate its schema, and submit it as V6 only after checking
the archive contents. Public score is not known yet.

Public artifact created: `submissions/legalir_public_v6_top50_keep3.zip`.
Validation: 1,000 queries, exactly 5 IDs per query, no missing corpus IDs, ZIP
contains only `submission.json`. SHA-256:
`841e4e3c103f4ea5c445035e595d89ed30ef619e547e71883b83373ad05afe31`.

## Rejected experiment: hard-negative reranker v2 (2026-08-13)

- 800 train queries, all gold documents, four hard negatives, 4,087 pairs.
- Fixed 100-query smoke: Recall 0.8058, Precision 0.1720, Hit@5 0.8500.
- V6 on the same smoke set: Recall 0.8508, Precision 0.1800, Hit@5 0.8900.

Decision: reject before full-dev inference. More data did not repair the
pointwise hard-negative objective; see `legalir_hardneg_v2_800.md`.

## Candidate V7: listwise LambdaLoss reranker (2026-08-13)

- 300 train queries with all positive documents and four mined hard negatives.
- Unlike the rejected pointwise models, this model ranks each query's candidate
  group jointly and optimizes LambdaLoss at top 5.
- Full dev with keep-base=4: Recall **0.8339**, Precision **0.1769**, Hit@5
  **0.8594**.
- V6 full dev: Recall 0.8264, Precision 0.1753, Hit@5 0.8524.

Decision: accept as a local V7 candidate (+0.0075 Recall, +0.0016 Precision).
Run a stability check before generating and submitting a Public archive; no V7
Public score exists yet. See `legalir_listwise_smoke300.md`.

## Rejected scale check: listwise 800 queries (2026-08-13)

- Fixed smoke100: Recall 0.8558, Precision 0.1820, Hit@5 0.9000.
- V7-300 on the same split has identical scores.

Decision: stop before full-dev inference. More listwise training queries did
not pass the local improvement gate; V7-300 remains the candidate.
