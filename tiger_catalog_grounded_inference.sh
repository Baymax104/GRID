#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="" SID="" EMBEDDING="" CHECKPOINT="" SPLIT="" GROUP="rkmeans" DEVICES="" ARM="full" NOTES="" DATASET=""
SEED=42 BEAM_WIDTH=10
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29730}
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --data-dir|--semantic-id-path|--embedding-path|--checkpoint-path|--data-split|--group|--devices|--arm|--notes|--seed|--master-port|--dataset|--beam-width)
      flag="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Error: $flag requires a non-empty value." >&2; exit 2
      fi
      value="$2"; shift 2 ;;
    --data-dir=*|--semantic-id-path=*|--embedding-path=*|--checkpoint-path=*|--data-split=*|--group=*|--devices=*|--arm=*|--notes=*|--seed=*|--master-port=*|--dataset=*|--beam-width=*)
      flag="${1%%=*}"; value="${1#*=}"; shift ;;
    --*) echo "Error: unknown option $1" >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift; continue ;;
  esac
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
  case "$flag" in
    --data-dir) DATA_DIR="$value" ;;
    --semantic-id-path) SID="$value" ;;
    --embedding-path) EMBEDDING="$value" ;;
    --checkpoint-path) CHECKPOINT="$value" ;;
    --data-split) SPLIT="$value" ;;
    --group) GROUP="$value" ;;
    --devices) DEVICES="$value" ;;
    --arm) ARM="$value" ;;
    --notes) NOTES="$value" ;;
    --seed) SEED="$value" ;;
    --master-port) MASTER_PORT="$value" ;;
    --dataset) DATASET="$value" ;;
    --beam-width) BEAM_WIDTH="$value" ;;
  esac
done
for value in "$DATA_DIR" "$SID" "$EMBEDDING" "$CHECKPOINT" "$SPLIT" "$DEVICES" "$DATASET"; do
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: data-dir, semantic-id-path, embedding-path, checkpoint-path, data-split, devices and dataset are required." >&2; exit 2; }
done
case "$ARM" in original|mask_ce|token_content_init|single_prototype|full|no_aux|shuffled|hybrid) ;;
  *) echo "Error: invalid catalog arm $ARM" >&2; exit 2 ;; esac
[[ "$SPLIT" == evaluation || "$SPLIT" == testing ]] || { echo "Error: data-split must be evaluation or testing." >&2; exit 2; }
[[ "$GROUP" == rkmeans || "$GROUP" == rvq ]] || { echo "Error: group must be rkmeans or rvq." >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Error: seed must be a nonnegative integer." >&2; exit 2; }
[[ "$BEAM_WIDTH" =~ ^[1-9][0-9]*$ ]] || { echo "Error: beam-width must be positive." >&2; exit 2; }
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Error: NPROC_PER_NODE must be positive." >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MASTER_PORT > 0 && 10#$MASTER_PORT <= 65535 )) || { echo "Error: invalid master port." >&2; exit 2; }

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
ARGS=(experiment=tiger_catalog_grounded_inference "data_dir=$(quote_hydra_string "$DATA_DIR")"
  "semantic_id_path=$(quote_hydra_string "$SID")" "embedding_path=$(quote_hydra_string "$EMBEDDING")"
  "ckpt_path=$(quote_hydra_string "$CHECKPOINT")" "checkpoint_reference=$(quote_hydra_string "$CHECKPOINT")"
  "dataset_name=$(quote_hydra_string "$DATASET")" "data_split=$SPLIT" "group=$GROUP" "devices=$DEVICES"
  "catalog_arm=$ARM" "seed=$SEED" "beam_width=$BEAM_WIDTH")
if [[ "$ARM" == hybrid ]]; then ARGS+=(prefix_trace=false callbacks.prefix_trace_writer=null); fi
if [[ -n "$NOTES" ]]; then ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")"); fi
if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
ARGS+=("${EXTRA_ARGS[@]}")

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE + 1 )) \
  uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m src.main "${ARGS[@]}"
