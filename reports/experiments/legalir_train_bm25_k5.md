# Experiment: LegalIR train BM25 baseline

## Data and method

- Train questions: 7,000
- Corpus: 8,532 context JSON files
- Retrieval: dependency-free BM25 over `passage`
- Submission cutoff: top 5 document IDs per query
- Purpose: establish the first real-data baseline before public submission

## Local metric by cutoff

```text
cutoff     Recall    Precision    hit@k
top-1      0.1680    0.1751       0.1751
top-3      0.2819    0.0988       0.2919
top-5      0.3468    0.0733       0.3584
```

The low precision is expected for a broad first-pass retriever returning five IDs per query; the competition's primary LegalIR metric is Recall, so this is a starting point, not a final candidate.

## Artifact

- Train prediction: `submissions/legalir_train_bm25_k5.json`
- Public prediction: `submissions/legalir_public_bm25_k5.json`

The public file has 1,000 predictions, each with exactly five IDs, and passed the local format validator. It has not been uploaded to Codabench.
