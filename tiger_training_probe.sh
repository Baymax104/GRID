#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="" SID="" INITIAL="" GROUP="" DEVICES="" ARM="" NOTES=""
SEED=42
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29630}
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --data-dir|--semantic-id-path|--initialization-checkpoint-path|--group|--devices|--arm|--notes|--seed|--master-port)
      flag="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Error: $flag requires a non-empty value." >&2; exit 2
      fi
      value="$2"; shift 2 ;;
    --data-dir=*|--semantic-id-path=*|--initialization-checkpoint-path=*|--group=*|--devices=*|--arm=*|--notes=*|--seed=*|--master-port=*)
      flag="${1%%=*}"; value="${1#*=}"; shift
      if [[ -z "$value" ]]; then echo "Error: $flag requires a non-empty value." >&2; exit 2; fi ;;
    --*) echo "Error: unknown option $1" >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift; continue ;;
  esac
  case "$flag" in
    --data-dir) DATA_DIR="$value" ;;
    --semantic-id-path) SID="$value" ;;
    --initialization-checkpoint-path) INITIAL="$value" ;;
    --group) GROUP="$value" ;;
    --devices) DEVICES="$value" ;;
    --arm) ARM="$value" ;;
    --notes) NOTES="$value" ;;
    --seed) SEED="$value" ;;
    --master-port) MASTER_PORT="$value" ;;
  esac
done

for value in "$DATA_DIR" "$SID" "$INITIAL" "$GROUP" "$DEVICES" "$ARM" "$NOTES"; do
  if [[ -z "${value//[[:space:]]/}" ]]; then
    echo "Error: data-dir, semantic-id-path, initialization-checkpoint-path, group, devices, arm and notes are required." >&2; exit 2
  fi
done
[[ "$ARM" == ce || "$ARM" == reweighted ]] || { echo "Error: arm must be ce or reweighted." >&2; exit 2; }
[[ "$GROUP" == rkmeans || "$GROUP" == rvq ]] || { echo "Error: group must be rkmeans or rvq." >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Error: seed must be non-negative integer." >&2; exit 2; }
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Error: NPROC_PER_NODE must be positive." >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MASTER_PORT > 0 && 10#$MASTER_PORT <= 65535 )) || { echo "Error: invalid master port." >&2; exit 2; }

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

ARGS=(experiment=tiger_training_probe "data_dir=$(quote_hydra_string "$DATA_DIR")"
  "semantic_id_path=$(quote_hydra_string "$SID")"
  "initialization_checkpoint_path=$(quote_hydra_string "$INITIAL")"
  "group=$GROUP" "devices=$DEVICES" "probe_arm=$ARM" "seed=$SEED"
  "logger.wandb.notes=$(quote_hydra_string "$NOTES")")
if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE + 1 )) \
  uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m src.main "${ARGS[@]}"
