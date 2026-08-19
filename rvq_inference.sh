#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

CKPT_PATH=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --ckpt-path=*)
      CKPT_PATH="${1#--ckpt-path=}"
      shift
      ;;
    --ckpt-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --ckpt-path requires a value." >&2
        exit 2
      fi
      CKPT_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CKPT_PATH" ]]; then
  echo "Error: --ckpt-path requires a value." >&2
  exit 2
fi

ARGS=(
  experiment=rvq_inference
  embedding_path=logs/sem_embeds_inference/runs/2026-08-06/11-30-14/pickle/merged_predictions_tensor.pt
  ckpt_path="$CKPT_PATH"
  devices=[0]
  data_dir=data/beauty
  embedding_dim=768
  num_hierarchies=3
  codebook_size=256
)

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.main \
  "${ARGS[@]}"
