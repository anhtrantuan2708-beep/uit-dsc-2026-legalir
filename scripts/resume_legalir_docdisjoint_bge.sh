#!/usr/bin/env bash
# Validate BGE corpus reranking on the full document-disjoint split.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUERIES="$ROOT/data/derived/legalir_docdisjoint_dev.json"
CORPUS="$ROOT/data/real/contexts"
CANDIDATES="$ROOT/tmp/docdisjoint_hybrid_w1_k20_top100.json"
cd "$ROOT"

mkdir -p tmp/docdisjoint_bge_shards
for start in 0 200 400 600 800 1000; do
  size=200
  [[ "$start" == 1000 ]] && size=8
  shard="tmp/docdisjoint_bge_shards/bge_${start}.json"
  if [[ -s "$shard" ]]; then
    echo "Reusing $shard"
    continue
  fi
  echo "Running document-disjoint BGE shard $start"
  .venv/bin/python src/rerank_legalir.py \
    "$QUERIES" "$CORPUS" "$CANDIDATES" "$shard" \
    --model BAAI/bge-reranker-v2-m3 \
    --model-cache models/bge-reranker-v2-m3 \
    --candidate-k 20 --top-k 20 --evidence-chunks 2 --batch-size 8 \
    --query-start "$start" --max-queries "$size"
done

.venv/bin/python src/merge_legalir_rankings.py \
  tmp/docdisjoint_bge_top20.json tmp/docdisjoint_bge_shards/bge_*.json
.venv/bin/python src/evaluate_legalir.py \
  "$QUERIES" tmp/docdisjoint_bge_top20.json --top-k 5
