# Experiment: sample-legalir-baseline

## Scope

- Dataset: `data/sample_legalir/` (synthetic practice data)
- Method: dependency-free lexical/BM25-style retrieval
- Candidate count: `top-k = 5`
- Purpose: verify the pipeline and submission contract, not estimate competition performance

## Result

```text
queries: 5
missing predictions: 0
hit@k: 1.0000
macro recall: 1.0000
macro precision: 0.4400
```

The k=5 result has lower precision because the baseline returns five documents while each synthetic query has one relevant document. A k=1 run on this deliberately easy sample produced Recall 1.0000 and Precision 1.0000.

## Reproduce

```bash
bash scripts/run_legalir_sample.sh
```

## Next swap

When the BTC supplies real LegalIR files, replace only:

```text
data/sample_legalir/queries.json  -> real question file
data/sample_legalir/contexts      -> real context_*.json directory
```

Keep the validator and evaluator, then compare local warmup results before any public submission.
