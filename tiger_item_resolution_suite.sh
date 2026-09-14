#!/usr/bin/env bash
# 两组人工启动队列；默认覆盖完整 54 次从头训练。
set -euo pipefail
QUEUE="" BEAUTY_DATA="" SPORTS_DATA="" NOTES=""
BEAUTY_SID=wandb://4vyi4o6w SPORTS_SID=wandb://3narllqy
BEAUTY_EMBED=wandb://3jtt9mpa SPORTS_EMBED=wandb://psec3u5i
SEEDS=42,2024,2025 ARMS=mir,earliest,depth2,depth_gate,dense,hybrid,cobra,mask_ce,token_content_init
DRY_RUN=false PRINT_ONLY=false EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --print-only) PRINT_ONLY=true; shift; continue ;;
    --queue|--beauty-data-dir|--sports-data-dir|--beauty-semantic-id-path|--sports-semantic-id-path|--beauty-embedding-path|--sports-embedding-path|--seeds|--arms|--notes)
      flag="$1"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
      value="$2"; shift 2 ;;
    --queue=*|--beauty-data-dir=*|--sports-data-dir=*|--beauty-semantic-id-path=*|--sports-semantic-id-path=*|--beauty-embedding-path=*|--sports-embedding-path=*|--seeds=*|--arms=*|--notes=*)
      flag="${1%%=*}"; value="${1#*=}"; shift ;;
    --*) echo "Error: unknown option $1" >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift; continue ;;
  esac
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
  case "$flag" in
    --queue) QUEUE="$value" ;;
    --beauty-data-dir) BEAUTY_DATA="$value" ;;
    --sports-data-dir) SPORTS_DATA="$value" ;;
    --beauty-semantic-id-path) BEAUTY_SID="$value" ;;
    --sports-semantic-id-path) SPORTS_SID="$value" ;;
    --beauty-embedding-path) BEAUTY_EMBED="$value" ;;
    --sports-embedding-path) SPORTS_EMBED="$value" ;;
    --seeds) SEEDS="$value" ;;
    --arms) ARMS="$value" ;;
    --notes) NOTES="$value" ;;
  esac
done
[[ "$QUEUE" == 1 || "$QUEUE" == 2 ]] || { echo "Error: queue must be 1 or 2." >&2; exit 2; }
[[ -n "$BEAUTY_DATA" && -n "$SPORTS_DATA" ]] || { echo "Error: both data directories are required." >&2; exit 2; }
[[ "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ && "$ARMS" =~ ^[a-z_0-9]+(,[a-z_0-9]+)*$ ]] || { echo "Error: seeds/arms must be nonempty comma-separated lists." >&2; exit 2; }
IFS=, read -r -a SEED_LIST <<< "$SEEDS"
IFS=, read -r -a ARM_LIST <<< "$ARMS"
declare -A SEEN_SEEDS=() SEEN_ARMS=()
for seed in "${SEED_LIST[@]}"; do
  [[ -z "${SEEN_SEEDS[$seed]:-}" ]] || { echo "Error: duplicate seed." >&2; exit 2; }
  SEEN_SEEDS[$seed]=1
done
for arm in "${ARM_LIST[@]}"; do
  case "$arm" in mir|earliest|depth2|depth_gate|dense|hybrid|cobra|mask_ce|token_content_init) ;;
    *) echo "Error: unknown arm $arm" >&2; exit 2 ;; esac
  [[ -z "${SEEN_ARMS[$arm]:-}" ]] || { echo "Error: duplicate arm." >&2; exit 2; }
  SEEN_ARMS[$arm]=1
done
for arg in "${EXTRA_ARGS[@]}"; do
  key="${arg%%=*}"; key="${key#+}"; key="${key#+}"
  case "$key" in seed|resolution_arm|dataset_name|data_dir|devices|trainer.root.devices|ckpt_path)
    echo "Error: use suite selectors instead of overriding matrix identity ($arg)." >&2; exit 2 ;;
  esac
done
if [[ "$QUEUE" == 1 ]]; then GPUS=0,1 PORT=29810; else GPUS=2,3 PORT=29820; fi
index=0
for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARM_LIST[@]}"; do
    for dataset in beauty sports; do
      assigned=$(((index + index / 2) % 2 + 1)); index=$((index + 1))
      [[ "$assigned" == "$QUEUE" ]] || continue
      if [[ "$dataset" == beauty ]]; then DATA="$BEAUTY_DATA" SID="$BEAUTY_SID" EMBED="$BEAUTY_EMBED";
      else DATA="$SPORTS_DATA" SID="$SPORTS_SID" EMBED="$SPORTS_EMBED"; fi
      ARGS=(--data-dir "$DATA" --dataset "$dataset" --semantic-id-path "$SID" --embedding-path "$EMBED"
        --arm "$arm" --seed "$seed" --devices '[0,1]' --master-port "$PORT"
        --notes "MIR complete method comparison; protocol=mir-v1; arm=$arm; dataset=$dataset; seed=$seed; from scratch; queue=$QUEUE; $NOTES")
      [[ "$DRY_RUN" == false ]] || ARGS+=(--dry-run)
      [[ "$PRINT_ONLY" == false ]] || ARGS+=(--print-only)
      ARGS+=("${EXTRA_ARGS[@]}")
      CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 bash ./tiger_item_resolution_train.sh "${ARGS[@]}"
    done
  done
done
