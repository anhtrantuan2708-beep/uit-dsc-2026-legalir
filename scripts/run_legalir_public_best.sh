#!/usr/bin/env bash
# Create the current best Public Test candidate. It does NOT upload anything.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/real/LegalIR - Public Test"

cd "$ROOT"

# 100 candidates per retriever are internal only. RRF reduces them to 5 IDs.
python3 src/legalir_baseline.py \
  "$DATA/public-official.json" "$ROOT/data/real/contexts" \
  "$ROOT/submissions/legalir_public_bm25_profiles_top100.json" \
  --include-title --fold-accents --remove-stopwords \
  --query-profiles "$ROOT/data/derived/legalir_all_train_profiles.json" \
  --top-k 100

.venv/bin/python src/dense_legalir.py \
  "$DATA/public-official.json" "$ROOT/data/real/contexts" \
  "$ROOT/submissions/legalir_public_dense_e5_top100.json" \
  --model-cache "$ROOT/models/e5-small" \
  --embeddings "$ROOT/data/derived/e5_corpus_embeddings.npy" \
  --embedding-ids "$ROOT/data/derived/e5_corpus_ids.json" \
  --top-k 100

.venv/bin/python src/knn_legalir.py \
  "$DATA/public-official.json" "$DATA/train.json" \
  "$ROOT/submissions/legalir_public_knn_train_questions_top100.json" \
  --model-cache "$ROOT/models/e5-small" \
  --embeddings "$ROOT/data/derived/e5_all_train_question_embeddings.npy" \
  --embedding-ids "$ROOT/data/derived/e5_all_train_question_ids.json" \
  --neighbors 100 --top-k 100

python3 src/fuse_legalir_multi.py \
  "$ROOT/submissions/legalir_public_v3_supervised_hybrid_k5.json" \
  "$ROOT/submissions/legalir_public_bm25_profiles_top100.json" \
  "$ROOT/submissions/legalir_public_dense_e5_top100.json" \
  "$ROOT/submissions/legalir_public_knn_train_questions_top100.json" \
  --weights 1,1,0.9 --rrf-k 15 --top-k 5

python3 src/validate_legalir_submission.py \
  "$DATA/public-official.json" \
  "$ROOT/submissions/legalir_public_v3_supervised_hybrid_k5.json"

cp "$ROOT/submissions/legalir_public_v3_supervised_hybrid_k5.json" "$ROOT/submissions/submission.json"
(cd "$ROOT/submissions" && zip -q -j legalir_public_v3_supervised_hybrid.zip submission.json)
unzip -t "$ROOT/submissions/legalir_public_v3_supervised_hybrid.zip"
echo "Ready to upload: $ROOT/submissions/legalir_public_v3_supervised_hybrid.zip"
