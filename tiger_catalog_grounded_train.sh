#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="" SID="" EMBEDDING="" GROUP="rkmeans" DEVICES="" ARM="full" NOTES="" DATASET=""
SEED=42
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29710}
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --data-dir|--semantic-id-path|--embedding-path|--group|--devices|--arm|--notes|--seed|--master-port|--dataset)
      flag="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Error: $flag requires a non-empty value." >&2; exit 2
      fi
      value="$2"; shift 2 ;;
    --data-dir=*|--semantic-id-path=*|--embedding-path=*|--group=*|--devices=*|--arm=*|--notes=*|--seed=*|--master-port=*|--dataset=*)
      flag="${1%%=*}"; value="${1#*=}"; shift ;;
    --*) echo "Error: unknown option $1" >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift; continue ;;
  esac
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
  case "$flag" in
    --data-dir) DATA_DIR="$value" ;;
    --semantic-id-path) SID="$value" ;;
    --embedding-path) EMBEDDING="$value" ;;
    --group) GROUP="$value" ;;
    --devices) DEVICES="$value" ;;
    --arm) ARM="$value" ;;
    --notes) NOTES="$value" ;;
    --seed) SEED="$value" ;;
    --master-port) MASTER_PORT="$value" ;;
    --dataset) DATASET="$value" ;;
  esac
done
for value in "$DATA_DIR" "$SID" "$EMBEDDING" "$DEVICES" "$DATASET"; do
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: data-dir, semantic-id-path, embedding-path, devices and dataset are required." >&2; exit 2; }
done
case "$ARM" in original|mask_ce|token_content_init|single_prototype|full|no_aux|shuffled|hybrid) ;;
  *) echo "Error: invalid catalog arm $ARM" >&2; exit 2 ;; esac
[[ "$GROUP" == rkmeans || "$GROUP" == rvq ]] || { echo "Error: group must be rkmeans or rvq." >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Error: seed must be a nonnegative integer." >&2; exit 2; }
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Error: NPROC_PER_NODE must be positive." >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MASTER_PORT > 0 && 10#$MASTER_PORT <= 65535 )) || { echo "Error: invalid master port." >&2; exit 2; }

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
ARGS=(experiment=tiger_catalog_grounded_train "data_dir=$(quote_hydra_string "$DATA_DIR")"
  "semantic_id_path=$(quote_hydra_string "$SID")" "embedding_path=$(quote_hydra_string "$EMBEDDING")"
  "dataset_name=$(quote_hydra_string "$DATASET")" "group=$GROUP" "devices=$DEVICES" "catalog_arm=$ARM" "seed=$SEED")
if [[ -n "$NOTES" ]]; then ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")"); fi
if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE + 1 )) \
  uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m src.main "${ARGS[@]}"
