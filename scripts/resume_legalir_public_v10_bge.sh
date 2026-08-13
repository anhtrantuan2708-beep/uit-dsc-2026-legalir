#!/usr/bin/env bash
# Resume only the expensive V10 BGE stage after V9 Public candidates exist.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC="$ROOT/data/real/LegalIR - Public Test/public-official.json"
CORPUS="$ROOT/data/real/contexts"
cd "$ROOT"

test -s tmp/public_v9_top50.json
mkdir -p tmp/public_v10_bge_shards tmp/legalir_v10_package

for start in $(seq 0 100 900); do
  shard="tmp/public_v10_bge_shards/bge_${start}.json"
  if [[ -s "$shard" ]]; then
    echo "Reusing $shard"
    continue
  fi
  echo "Running V10 BGE shard $start"
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
  tmp/public_v10_bge_top10.json submissions/legalir_public_v10_k5.json \
  --keep-base 3 --top-k 5
.venv/bin/python src/validate_legalir_submission.py \
  "$PUBLIC" submissions/legalir_public_v10_k5.json

cp submissions/legalir_public_v10_k5.json tmp/legalir_v10_package/submission.json
(cd tmp/legalir_v10_package && zip -q -FS "$ROOT/submissions/legalir_public_v10.zip" submission.json)
unzip -t submissions/legalir_public_v10.zip
shasum -a 256 submissions/legalir_public_v10.zip
echo "Ready: $ROOT/submissions/legalir_public_v10.zip"
