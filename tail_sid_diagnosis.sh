#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/beauty}"
SEMANTIC_ID_PATH="${SEMANTIC_ID_PATH:-logs/rkmeans_inference/runs/2026-08-14/10-33-53/pickle/merged_predictions_tensor.pt}"
EMBEDDING_PATH="${EMBEDDING_PATH:-logs/sem_embeds_inference/runs/2026-08-06/11-30-14/pickle/merged_predictions_tensor.pt}"
RAW_NUM_HIERARCHIES="${RAW_NUM_HIERARCHIES:-3}"

ARGS=(
  experiment=tail_sid_diagnosis
  data_dir="$DATA_DIR"
  semantic_id_path="$SEMANTIC_ID_PATH"
  raw_num_hierarchies="$RAW_NUM_HIERARCHIES"
)

if [[ "$EMBEDDING_PATH" != "null" ]]; then
  ARGS+=(embedding_path="$EMBEDDING_PATH")
fi

uv run --module src.main "${ARGS[@]}"
