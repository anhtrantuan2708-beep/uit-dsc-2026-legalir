#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/real/LegalIR - Public Test"
PROFILES="$ROOT/data/derived/legalir_all_train_profiles.json"
OUTPUT="$ROOT/submissions/legalir_public_bm25_profiles_k5.json"

cd "$ROOT"
mkdir -p submissions

python3 src/build_legalir_query_profiles.py \
  "$DATA/train.json" \
  "$PROFILES"

python3 src/legalir_baseline.py \
  "$DATA/public-official.json" \
  "$ROOT/data/real/contexts" \
  "$OUTPUT" \
  --top-k 5 \
  --include-title \
  --fold-accents \
  --remove-stopwords \
  --query-profiles "$PROFILES"

python3 src/validate_legalir_submission.py \
  "$DATA/public-official.json" \
  "$OUTPUT"

echo "Ready to package: $OUTPUT"
