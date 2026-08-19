#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGS=(
  experiment=sem_embeds_inference
  devices=[0]
  data_dir=data/beauty
)

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.main \
  "${ARGS[@]}"
