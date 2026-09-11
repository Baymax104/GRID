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
ALLOCATION_ENABLED=""
RESERVED_SLOTS=""
POOL_MULTIPLIER=""
SOURCE_SPLIT="training"
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
    --data-dir=*|--data-split=*|--beam-width=*|--seed=*|--master-port=*|--devices=*|--group=*|--ckpt-path=*|--semantic-id-path=*|--notes=*|--allocation-enabled=*|--reserved-slots=*|--pool-multiplier=*|--source-split=*)
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
        --allocation-enabled) ALLOCATION_ENABLED="$value" ;;
        --reserved-slots) RESERVED_SLOTS="$value" ;;
        --pool-multiplier) POOL_MULTIPLIER="$value" ;;
        --source-split) SOURCE_SPLIT="$value" ;;
      esac
      shift
      ;;
    --data-dir|--data-split|--beam-width|--seed|--master-port|--devices|--group|--ckpt-path|--semantic-id-path|--notes|--allocation-enabled|--reserved-slots|--pool-multiplier|--source-split)
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
        --allocation-enabled) ALLOCATION_ENABLED="$value" ;;
        --reserved-slots) RESERVED_SLOTS="$value" ;;
        --pool-multiplier) POOL_MULTIPLIER="$value" ;;
        --source-split) SOURCE_SPLIT="$value" ;;
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
require_value --group "$GROUP"
require_value --notes "$NOTES"
require_value --allocation-enabled "$ALLOCATION_ENABLED"
require_value --reserved-slots "$RESERVED_SLOTS"
require_value --pool-multiplier "$POOL_MULTIPLIER"

case "$DATA_SPLIT" in evaluation|testing) ;; *) echo "Error: --data-split must be evaluation or testing." >&2; exit 2 ;; esac
case "$SOURCE_SPLIT" in training) ;; *) echo "Error: --source-split must be training." >&2; exit 2 ;; esac
case "$GROUP" in rkmeans|rvq|rqvae) ;; *) echo "Error: --group requires one of: rkmeans, rvq, rqvae." >&2; exit 2 ;; esac
case "$ALLOCATION_ENABLED" in true|false) ;; *) echo "Error: --allocation-enabled must be true or false." >&2; exit 2 ;; esac
if ! [[ "$BEAM_WIDTH" =~ ^[1-9][0-9]*$ ]]; then echo "Error: --beam-width must be a positive integer." >&2; exit 2; fi
if ! [[ "$RESERVED_SLOTS" =~ ^[0-9]+$ ]]; then echo "Error: --reserved-slots must be a non-negative integer." >&2; exit 2; fi
if ! [[ "$POOL_MULTIPLIER" =~ ^[1-9][0-9]*$ ]]; then echo "Error: --pool-multiplier must be a positive integer." >&2; exit 2; fi
if [[ "$ALLOCATION_ENABLED" == true ]]; then
  if (( RESERVED_SLOTS <= 0 || RESERVED_SLOTS >= BEAM_WIDTH )); then
    echo "Error: enabled allocation requires 0 < --reserved-slots < --beam-width." >&2
    exit 2
  fi
elif (( RESERVED_SLOTS != 0 )); then
  echo "Error: disabled allocation requires --reserved-slots=0." >&2
  exit 2
fi

ARGS=(
  experiment=tiger_prefix_allocation_probe
  group="$GROUP"
  data_dir="$DATA_DIR"
  data_split="$DATA_SPLIT"
  beam_width="$BEAM_WIDTH"
  seed="$SEED"
  devices="$DEVICES"
  ckpt_path="$CKPT_PATH"
  checkpoint_reference="$CKPT_PATH"
  semantic_id_path="$SEMANTIC_ID_PATH"
  prefix_allocation.enabled="$ALLOCATION_ENABLED"
  prefix_allocation.reserved_slots="$RESERVED_SLOTS"
  prefix_allocation.pool_multiplier="$POOL_MULTIPLIER"
  prefix_allocation.source_split="$SOURCE_SPLIT"
  "logger.wandb.notes=$(quote_hydra_string "$NOTES")"
)
if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
ARGS+=("${EXTRA_ARGS[@]}")

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE")
if [[ -n "$MASTER_PORT" ]]; then TORCHRUN_ARGS+=(--master_port="$MASTER_PORT"); fi

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run torchrun "${TORCHRUN_ARGS[@]}" -m src.main "${ARGS[@]}"
