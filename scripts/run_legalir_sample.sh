#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p submissions reports/experiments

python3 src/legalir_baseline.py \
  data/sample_legalir/queries.json \
  data/sample_legalir/contexts \
  submissions/sample_legalir_k5.json \
  --top-k 5

python3 src/validate_legalir_submission.py \
  data/sample_legalir/queries.json \
  submissions/sample_legalir_k5.json

python3 src/evaluate_legalir.py \
  data/sample_legalir/queries.json \
  submissions/sample_legalir_k5.json
