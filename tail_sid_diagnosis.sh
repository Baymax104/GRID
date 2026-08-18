#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/beauty}"
SEMANTIC_ID_PATH="${SEMANTIC_ID_PATH:-logs/rkmeans_inference/runs/2026-08-14/10-33-53/pickle/merged_predictions_tensor.pt}"
EMBEDDING_PATH="${EMBEDDING_PATH:-logs/sem_embeds_inference/runs/2026-08-06/11-30-14/pickle/merged_predictions_tensor.pt}"
RAW_NUM_HIERARCHIES="${RAW_NUM_HIERARCHIES:-3}"

NOTES=""
DRY_RUN=false
EXTRA_ARGS=()

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --notes=*)
      NOTES="${1#--notes=}"
      shift
      ;;
    --notes)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --notes requires a value." >&2
        exit 2
      fi
      NOTES="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  experiment=tail_sid_diagnosis
  data_dir="$DATA_DIR"
  semantic_id_path="$SEMANTIC_ID_PATH"
  raw_num_hierarchies="$RAW_NUM_HIERARCHIES"
)

if [[ "$EMBEDDING_PATH" != "null" ]]; then
  ARGS+=(embedding_path="$EMBEDDING_PATH")
fi

if [[ -n "$NOTES" ]]; then
  ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")
fi

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

uv run --module src.main "${ARGS[@]}"
