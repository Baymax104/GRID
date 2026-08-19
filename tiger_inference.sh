#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

CKPT_PATH=""
SEMANTIC_ID_PATH=""
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
    --semantic-id-path=*)
      SEMANTIC_ID_PATH="${1#--semantic-id-path=}"
      shift
      ;;
    --semantic-id-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --semantic-id-path requires a value." >&2
        exit 2
      fi
      SEMANTIC_ID_PATH="$2"
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

if [[ -z "$SEMANTIC_ID_PATH" ]]; then
  echo "Error: --semantic-id-path requires a value." >&2
  exit 2
fi

ARGS=(
  experiment=tiger_inference
  ckpt_path="$CKPT_PATH"
  semantic_id_path="$SEMANTIC_ID_PATH"
  devices=[0]
  data_dir=data/beauty
  num_hierarchies=4
  codebook_size=256
  embedding_dim=256
)

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node="$NPROC_PER_NODE" -m src.main \
  "${ARGS[@]}"
