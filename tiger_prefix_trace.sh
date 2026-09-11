#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICES="${DEVICES:-[0]}"
DATA_DIR=""
DATA_SPLIT=""
BEAM_WIDTH=""
SEED="42"
CKPT_PATH=""
SEMANTIC_ID_PATH=""
GROUP=""
NOTES=""
DRY_RUN=false
EXTRA_ARGS=()

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Error: $option requires a value." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --data-dir=*|--data-split=*|--beam-width=*|--seed=*|--master-port=*|--devices=*|--group=*|--ckpt-path=*|--semantic-id-path=*|--notes=*)
      option="${1%%=*}"
      value="${1#*=}"
      require_value "$option" "$value"
      case "$option" in
        --data-dir) DATA_DIR="$value" ;;
        --data-split) DATA_SPLIT="$value" ;;
        --beam-width) BEAM_WIDTH="$value" ;;
        --seed) SEED="$value" ;;
        --master-port) MASTER_PORT="$value" ;;
        --devices) DEVICES="$value" ;;
        --group) GROUP="$value" ;;
        --ckpt-path) CKPT_PATH="$value" ;;
        --semantic-id-path) SEMANTIC_ID_PATH="$value" ;;
        --notes) NOTES="$value" ;;
      esac
      shift
      ;;
    --data-dir|--data-split|--beam-width|--seed|--master-port|--devices|--group|--ckpt-path|--semantic-id-path|--notes)
      require_value "$1" "${2:-}"
      option="$1"
      value="$2"
      case "$option" in
        --data-dir) DATA_DIR="$value" ;;
        --data-split) DATA_SPLIT="$value" ;;
        --beam-width) BEAM_WIDTH="$value" ;;
        --seed) SEED="$value" ;;
        --master-port) MASTER_PORT="$value" ;;
        --devices) DEVICES="$value" ;;
        --group) GROUP="$value" ;;
        --ckpt-path) CKPT_PATH="$value" ;;
        --semantic-id-path) SEMANTIC_ID_PATH="$value" ;;
        --notes) NOTES="$value" ;;
      esac
      shift 2
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

require_value --data-dir "$DATA_DIR"
require_value --data-split "$DATA_SPLIT"
require_value --beam-width "$BEAM_WIDTH"
require_value --seed "$SEED"
require_value --devices "$DEVICES"
require_value --ckpt-path "$CKPT_PATH"
require_value --semantic-id-path "$SEMANTIC_ID_PATH"
require_value --notes "$NOTES"

case "$DATA_SPLIT" in evaluation|testing) ;; *) echo "Error: --data-split must be evaluation or testing." >&2; exit 2 ;; esac
case "$GROUP" in
  rkmeans|rvq|rqvae) ;;
  "") echo "Error: --group requires one of: rkmeans, rvq, rqvae." >&2; exit 2 ;;
  *) echo "Error: unsupported --group '$GROUP'; expected one of: rkmeans, rvq, rqvae." >&2; exit 2 ;;
esac
if ! [[ "$BEAM_WIDTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --beam-width must be a positive integer." >&2
  exit 2
fi

ARGS=(
  experiment=tiger_prefix_trace
  group="$GROUP"
  data_dir="$DATA_DIR"
  data_split="$DATA_SPLIT"
  beam_width="$BEAM_WIDTH"
  seed="$SEED"
  devices="$DEVICES"
  ckpt_path="$CKPT_PATH"
  semantic_id_path="$SEMANTIC_ID_PATH"
  "logger.wandb.notes=$(quote_hydra_string "$NOTES")"
)
if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
ARGS+=("${EXTRA_ARGS[@]}")

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE")
if [[ -n "$MASTER_PORT" ]]; then TORCHRUN_ARGS+=(--master_port="$MASTER_PORT"); fi

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run torchrun "${TORCHRUN_ARGS[@]}" -m src.main "${ARGS[@]}"
