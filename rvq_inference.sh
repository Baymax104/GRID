#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICES="${DEVICES:-[0]}"

CKPT_PATH=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --master-port=*)
      MASTER_PORT="${1#--master-port=}"
      shift
      ;;
    --master-port)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --master-port requires a value." >&2
        exit 2
      fi
      MASTER_PORT="$2"
      shift 2
      ;;
    --devices=*)
      DEVICES="${1#--devices=}"
      shift
      ;;
    --devices)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --devices requires a value." >&2
        exit 2
      fi
      DEVICES="$2"
      shift 2
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
  devices="$DEVICES"
  data_dir=data/beauty
)

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE")
if [[ -n "$MASTER_PORT" ]]; then
  TORCHRUN_ARGS+=(--master_port="$MASTER_PORT")
fi

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun "${TORCHRUN_ARGS[@]}" -m src.main \
  "${ARGS[@]}"
