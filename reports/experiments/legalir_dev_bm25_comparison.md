# LegalIR validation — BM25 comparison

Validation split: 1,003 questions drawn deterministically from the 7,000 labelled training questions. Corpus: 8,532 passages. Both systems return top 5 IDs.

| Variant | Recall | Precision | Hit@5 |
|---|---:|---:|---:|
| Raw BM25 over `passage` | 0.3431 | 0.0724 | 0.3549 |
| Normalized BM25 over `name + passage`, with stopwords removed | 0.4189 | 0.0889 | 0.4347 |
| Normalized BM25 + document query profiles from train split | 0.4234 | 0.0899 | 0.4397 |

The profile variant improves Recall by 0.0803 (23.4% relative) over raw BM25 and is the current lexical baseline selected for Public Test generation.

This is still well below the public leaderboard (~0.93–0.96 Recall), so the next research stage is a local dense retriever and hybrid fusion with this BM25 model.
