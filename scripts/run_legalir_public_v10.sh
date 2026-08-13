#!/usr/bin/env bash
# Reproduce the V10 Public candidate locally. This script never uploads it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/real/LegalIR - Public Test"
PUBLIC="$DATA/public-official.json"
TRAIN="$DATA/train.json"
CORPUS="$ROOT/data/real/contexts"

cd "$ROOT"

.venv/bin/python src/char_ngram_question_knn_legalir.py \
  "$PUBLIC" "$TRAIN" tmp/public_char_ngram_question_knn_top100.json \
  --neighbors 100 --top-k 100

.venv/bin/python src/fuse_legalir_rankings.py \
  submissions/legalir_public_v4_fourway_top100.json \
  tmp/public_char_ngram_question_knn_top100.json \
  tmp/public_v8_candidates_top100.json \
  --first-weight 1 --second-weight 0.35 --rrf-k 20 --top-k 100

.venv/bin/python src/rerank_legalir.py \
  "$PUBLIC" "$CORPUS" \
  tmp/public_v8_candidates_top100.json tmp/public_v8_listwise_top50.json \
  --model models/mmarco-legalir-listwise-smoke300 \
  --candidate-k 50 --top-k 50 --evidence-chunks 2 --batch-size 24

.venv/bin/python src/knn_legalir.py \
  "$PUBLIC" "$TRAIN" tmp/public_dense_question_knn_tuned_top100.json \
  --embeddings data/derived/e5_all_train_question_embeddings.npy \
  --embedding-ids data/derived/e5_all_train_question_ids.json \
  --query-embeddings data/derived/e5_public_question_embeddings.npy \
  --query-embedding-ids data/derived/e5_public_question_ids.json \
  --neighbors 50 --top-k 100 --aggregation rank_vote \
  --similarity-power 4 --rank-power 0.5

.venv/bin/python src/fuse_legalir_rankings.py \
  tmp/public_v8_listwise_top50.json \
  tmp/public_dense_question_knn_tuned_top100.json \
  tmp/public_v9_top50.json \
  --first-weight 1 --second-weight 0.5 --rrf-k 1 --top-k 50

mkdir -p tmp/public_v10_bge_shards
for start in $(seq 0 100 900); do
  shard="tmp/public_v10_bge_shards/bge_${start}.json"
  if [[ -s "$shard" ]]; then
    echo "Reusing completed BGE shard: $shard"
    continue
  fi
  .venv/bin/python src/rerank_legalir.py \
    "$PUBLIC" "$CORPUS" tmp/public_v9_top50.json "$shard" \
    --model BAAI/bge-reranker-v2-m3 \
    --model-cache models/bge-reranker-v2-m3 \
    --candidate-k 10 --top-k 10 --evidence-chunks 2 --batch-size 8 \
    --query-start "$start" --max-queries 100
done

.venv/bin/python src/merge_legalir_rankings.py \
  tmp/public_v10_bge_top10.json tmp/public_v10_bge_shards/bge_*.json

.venv/bin/python src/blend_legalir_rankings.py \
  submissions/legalir_public_v5_safe_rerank_k5.json \
  tmp/public_v10_bge_top10.json \
  submissions/legalir_public_v10_k5.json \
  --keep-base 3 --top-k 5

.venv/bin/python src/validate_legalir_submission.py \
  "$PUBLIC" submissions/legalir_public_v10_k5.json

rm -rf tmp/legalir_v10_package
mkdir -p tmp/legalir_v10_package
cp submissions/legalir_public_v10_k5.json tmp/legalir_v10_package/submission.json
rm -f submissions/legalir_public_v10.zip
(cd tmp/legalir_v10_package && zip -q "$ROOT/submissions/legalir_public_v10.zip" submission.json)
unzip -t submissions/legalir_public_v10.zip
shasum -a 256 submissions/legalir_public_v10.zip
echo "Ready to upload: $ROOT/submissions/legalir_public_v10.zip"
