#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/real/LegalIR - Public Test"

cd "$ROOT"
bash scripts/run_legalir_public_baseline.sh

.venv/bin/python src/dense_legalir.py \
  "$DATA/public-official.json" \
  "$ROOT/data/real/contexts" \
  "$ROOT/submissions/legalir_public_dense_e5_k5.json" \
  --model-cache "$ROOT/models/e5-small" \
  --embeddings "$ROOT/data/derived/e5_corpus_embeddings.npy" \
  --embedding-ids "$ROOT/data/derived/e5_corpus_ids.json" \
  --top-k 5

python3 src/fuse_legalir_rankings.py \
  "$ROOT/submissions/legalir_public_bm25_profiles_k5.json" \
  "$ROOT/submissions/legalir_public_dense_e5_k5.json" \
  "$ROOT/submissions/legalir_public_hybrid_rrf_k5.json" \
  --top-k 5

python3 src/validate_legalir_submission.py \
  "$DATA/public-official.json" \
  "$ROOT/submissions/legalir_public_hybrid_rrf_k5.json"
