#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

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
  experiment=rvq_train
  embedding_path=logs/sem_embeds_inference/runs/2026-08-06/11-30-14/pickle/merged_predictions_tensor.pt
  data_dir=data/beauty
  devices=[0,1]
  embedding_dim=768
  num_hierarchies=3
  codebook_size=256
)

if [[ -n "$NOTES" ]]; then
  ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")
fi

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.main \
  "${ARGS[@]}"
