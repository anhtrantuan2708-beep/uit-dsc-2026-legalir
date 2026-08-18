#!/usr/bin/env bash
# Resume full-dev validation of the confidence-routed chunk reranker.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p tmp/v12_chunk_dev_shards
for start in 0 100 200 300 400 500 600 700 800 900 1000; do
  size=100
  [[ "$start" == 1000 ]] && size=3
  ranking="tmp/v12_chunk_dev_shards/ranking_${start}.json"
  scores="tmp/v12_chunk_dev_shards/scores_${start}.json"
  if [[ -s "$ranking" && -s "$scores" ]]; then
    echo "Reusing shard $start"
    continue
  fi
  .venv/bin/python src/rerank_fts_chunks_legalir.py \
    data/derived/legalir_dev.json \
    data/derived/legalir_chunks_fts5.sqlite \
    "$ranking" \
    --scores-output "$scores" \
    --chunk-k 1000 --candidate-k 20 --top-k 20 --batch-size 16 \
    --query-start "$start" --max-queries "$size"
done

.venv/bin/python src/merge_legalir_rankings.py \
  tmp/dev_v12_fts_chunkaware_bge_top20.json \
  tmp/v12_chunk_dev_shards/ranking_*.json
.venv/bin/python src/merge_legalir_rankings.py \
  tmp/dev_v12_fts_chunkaware_bge_scores.json \
  tmp/v12_chunk_dev_shards/scores_*.json
.venv/bin/python src/route_confident_legalir.py \
  tmp/dev_v9_bge_top10_w0_keep3.json \
  tmp/dev_v12_fts_chunkaware_bge_top20.json \
  tmp/dev_v12_fts_chunkaware_bge_scores.json \
  tmp/dev_v12_chunk_router_k5.json \
  --min-score 0.95 --min-margin 0 --keep-base 4 --top-k 5
.venv/bin/python src/evaluate_legalir.py \
  data/derived/legalir_dev.json tmp/dev_v12_chunk_router_k5.json --top-k 5
