# LegalIR V12 chunk retrieval and confidence router

Date: 2026-08-17

## Motivation

The earlier corpus retrievers indexed each legal document as one long passage.
This hides relevant Điều/Chương text deep inside documents. V12 builds a
disk-backed SQLite FTS5 index over 463,253 legal chunks while preserving each
source document ID.

## Chunk candidate retrieval

On the fixed 100-query smoke block, direct FTS chunk retrieval reached:

| Cutoff | Recall |
|---|---:|
| 5 | 0.7250 |
| 20 | 0.8367 |
| 50 | 0.8992 |
| 100 | **0.9467** |

The candidate ceiling is strong, but fixed RRF fusion and unconditional BGE
reranking both reduced V10 Recall. They are rejected.

## Confidence-routed fallback

BGE reranks the actual matched chunks rather than reconstructed whole-document
evidence. The alternate ranking is used only when:

- BGE top score is at least 0.93;
- the first four V10 IDs are retained, and only the fifth slot is filled from
  the chunk ranking.

The initial `0.9 / 0.002` confidence rule passed smoke blocks A–C. A later
threshold check removed the margin gate. On the first 900 full-Dev queries,
the `0.93 / 0` gate reached Recall 0.8683, narrowly above `0.95 / 0` at
0.8678; both retain the first four V10 IDs and route only the fifth slot. The
rule is therefore fixed as `0.93 / 0` for final full-Dev validation.

| Dev scope | V10 Recall@5 | V12 Recall@5 | V10 Precision@5 | V12 Precision@5 |
|---|---:|---:|---:|---:|
| First 900 queries | 0.8575 | **0.8683** | 0.1818 | **0.1847** |

The earlier smoke result with the initial conservative rule was:

| Block | V10 Recall@5 | Router Recall@5 | Gain | Router Precision@5 |
|---|---:|---:|---:|---:|
| A | 0.8692 | 0.8817 | +0.0125 | 0.1900 |
| B | 0.8750 | 0.8950 | +0.0200 | 0.1840 |
| C | 0.8000 | 0.8350 | **+0.0350** | 0.1720 |

The repeated gain passes the smoke gate. Full 1,003-query Dev validation is
the next required gate. It is resumable in 100-query shards through
`scripts/resume_legalir_v12_chunk_router_dev.sh`. No Public submission exists
for this candidate yet, and V10 remains the confirmed Public baseline.

## Rejected V13 wide-candidate check

Reranking 50 FTS candidates instead of 20 looked slightly better on Dev
queries 900–999 (Recall 0.8583 at the pre-fixed router setting, versus V12
0.8533), but it failed the independent 800–899 block: V13 reached 0.8300,
the same as V12 and below V10 at 0.8650. It is therefore rejected rather than
scaled to full Dev. The next improvement must make the chunk reranker itself
more discriminative, not simply provide it more candidates.

## Rejected V14 chunk fine-tuning check

`finetune_fts_chunk_crossencoder.py` was verified end-to-end with 800 labelled
Train questions: the positive example is the best lexical chunk of a gold
document and two FTS-returned non-gold chunks are hard negatives. On a held-out
100-query Dev block, the fine-tuned model alone scored Recall 0.3508, versus
0.8217 for V10; confidence routing could only recover the V10 baseline by not
using it. The weakly selected positive chunk is not sufficient supervision for
this model, so this branch is rejected and is not scaled.

## Data representation audit (2026-08-18)

The official corpus contains 8,532 documents, but its raw representation is
very uneven: median passage length is 23,110 characters, p99 is 285,366, and
the longest passage is 5.98 million characters. In addition, 1,125 documents
have no populated `name` field. Searching a complete document therefore gives
the model a large amount of unrelated text.

`build_legalir_metadata.py` derives a local sidecar only from the organizer
corpus: document number (8,253 documents), legal type (8,199), title (8,281)
and issuer (8,499). The sidecar is a useful fallback for explicit legal
references, but a compact metadata-only FTS test was weak on smoke100
(Recall@5 0.2500; Recall@100 0.5317), so it must not replace V10 or the chunk
retriever. `chunk_legalir_contexts.py --structured` additionally retains the
Article heading when it splits by Clause/Point. It is an implementation-ready
next data representation, but it is **not** part of the current V12 full-Dev
run: that run validates the existing Article-level chunk index and its
matched-chunk reranker first.

## Rejected BGE question-KNN check

BGE reranking of E5-nearest labelled questions improved one smoke block from
0.8692 to 0.8817, but the gain disappeared on full Dev. This branch is kept as
reproducible source code but is not a Public candidate.
