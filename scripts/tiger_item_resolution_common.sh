#!/usr/bin/env bash
# 由根 train/inference/calibration 入口 source；不直接运行实验。
set -euo pipefail
[[ -f src/main.py ]] || { echo "Error: run from the GRID repository root." >&2; exit 2; }
DATA_DIR="" SID="" EMBEDDING="" CHECKPOINT="" DATASET="" NOTES="" CALIBRATION=""
ARM=mir GROUP=rkmeans SEED=42 DEVICES='[0]' GPU=0 SPLIT=evaluation POLICY=standard
MASTER_PORT=${MASTER_PORT:-29810}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
DRY_RUN=false PRINT_ONLY=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --print-only) PRINT_ONLY=true; shift; continue ;;
    --data-dir|--semantic-id-path|--embedding-path|--checkpoint-path|--dataset|--notes|--arm|--group|--seed|--devices|--gpu|--data-split|--policy|--calibration-path|--master-port)
      flag="$1"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
      value="$2"; shift 2 ;;
    --data-dir=*|--semantic-id-path=*|--embedding-path=*|--checkpoint-path=*|--dataset=*|--notes=*|--arm=*|--group=*|--seed=*|--devices=*|--gpu=*|--data-split=*|--policy=*|--calibration-path=*|--master-port=*)
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
    --dataset) DATASET="$value" ;;
    --notes) NOTES="$value" ;;
    --arm) ARM="$value" ;;
    --group) GROUP="$value" ;;
    --seed) SEED="$value" ;;
    --devices) DEVICES="$value" ;;
    --gpu) GPU="$value" ;;
    --data-split) SPLIT="$value" ;;
    --policy) POLICY="$value" ;;
    --calibration-path) CALIBRATION="$value" ;;
    --master-port) MASTER_PORT="$value" ;;
  esac
done
for value in "$DATA_DIR" "$SID" "$EMBEDDING" "$DATASET"; do
  [[ -n "$value" ]] || { echo "Error: data-dir, dataset, semantic-id-path and embedding-path are required." >&2; exit 2; }
done
case "$ARM" in mir|earliest|depth2|depth_gate|dense|hybrid|cobra|mask_ce|token_content_init) ;;
  *) echo "Error: unknown arm $ARM" >&2; exit 2 ;; esac
[[ "$GROUP" == rkmeans || "$GROUP" == rvq ]] || { echo "Error: group must be rkmeans or rvq." >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ && "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Error: invalid seed or process count." >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MASTER_PORT > 0 && 10#$MASTER_PORT <= 65535 )) || { echo "Error: invalid master port." >&2; exit 2; }
quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
ARGS=("experiment=tiger_item_resolution_$MODE" "data_dir=$(quote_hydra_string "$DATA_DIR")"
  "semantic_id_path=$(quote_hydra_string "$SID")" "embedding_path=$(quote_hydra_string "$EMBEDDING")"
  "dataset_name=$(quote_hydra_string "$DATASET")" "group=$GROUP" "resolution_arm=$ARM" "seed=$SEED")
if [[ "$MODE" == train ]]; then
  ARGS+=("devices=$DEVICES")
  COMMAND=(uv run torchrun "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m src.main)
else
  [[ "$NPROC_PER_NODE" == 1 && ( "$DEVICES" == '[0]' || "$DEVICES" == 1 ) && "$GPU" =~ ^[0-9]+$ ]] || { echo "Error: inference/calibration requires one GPU and one process; use --gpu." >&2; exit 2; }
  [[ -n "$CHECKPOINT" ]] || { echo "Error: checkpoint-path is required." >&2; exit 2; }
  if [[ "$MODE" == calibration ]]; then
    ARM=hybrid POLICY=calibrate SPLIT=training
  else
    [[ "$SPLIT" == evaluation || "$SPLIT" == testing ]] || { echo "Error: data-split must be evaluation or testing." >&2; exit 2; }
    [[ "$POLICY" == standard || "$POLICY" == wide ]] || { echo "Error: invalid inference policy." >&2; exit 2; }
  fi
  if [[ "$POLICY" == wide ]]; then
    [[ "$ARM" == hybrid && -n "$CALIBRATION" ]] || { echo "Error: WIDE requires hybrid and calibration-path." >&2; exit 2; }
    ARGS+=("calibration_path=$(quote_hydra_string "$CALIBRATION")" search_states=4096 search_item_scores=4096 data.predict_dataloader.batch_size_per_device=8)
  fi
  ARGS+=("ckpt_path=$(quote_hydra_string "$CHECKPOINT")" "checkpoint_reference=$(quote_hydra_string "$CHECKPOINT")"
    "resolution_arm=$ARM" "inference_policy=$POLICY" "data_split=$SPLIT" 'devices=[0]' trainer.root.num_nodes=1)
  # 原生覆盖仍可调整模型参数，但不能绕过用户规定的单卡边界。
  for arg in "${EXTRA_ARGS[@]}"; do
    key="${arg%%=*}"; key="${key#+}"; key="${key#+}"
    value="${arg#*=}"
    case "$key" in
      devices|trainer.root.devices)
        [[ "$value" == '[0]' || "$value" == 1 ]] || { echo "Error: multi-GPU inference override is forbidden." >&2; exit 2; } ;;
      trainer.root.num_nodes)
        [[ "$value" == 1 ]] || { echo "Error: multi-node inference is forbidden." >&2; exit 2; } ;;
      trainer.root.strategy)
        [[ "$value" == auto ]] || { echo "Error: inference strategy must be auto." >&2; exit 2; } ;;
    esac
  done
  export CUDA_VISIBLE_DEVICES="$GPU"
  COMMAND=(uv run --module src.main)
fi
[[ -z "$NOTES" ]] || ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")
[[ "$DRY_RUN" == false ]] || ARGS+=(--dry-run)
ARGS+=("${EXTRA_ARGS[@]}")
if [[ "$PRINT_ONLY" == true ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"; fi
  printf '%q ' "${COMMAND[@]}" "${ARGS[@]}"
  printf '\n'
else
  "${COMMAND[@]}" "${ARGS[@]}"
fi
