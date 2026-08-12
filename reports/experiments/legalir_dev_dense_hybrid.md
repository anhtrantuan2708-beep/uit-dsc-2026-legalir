# LegalIR validation — Dense retrieval and hybrid fusion

Validation split: 1,003 questions. Corpus: 8,532 passages.

| Variant | Recall | Precision | Hit@5 |
|---|---:|---:|---:|
| BM25 + title + normalization + query profiles | 0.4234 | 0.0899 | 0.4397 |
| Dense: `intfloat/multilingual-e5-small` | 0.4688 | 0.0985 | 0.4855 |
| Hybrid: BM25 + dense Reciprocal Rank Fusion | **0.5571** | **0.1174** | **0.5773** |

The dense model runs locally. The corpus was embedded once and cached in `data/derived/e5_corpus_embeddings.npy`; no task data was sent to an AI API.

The hybrid is the current selected candidate. It improves Recall by 0.1337 over the best lexical baseline (31.6% relative).
