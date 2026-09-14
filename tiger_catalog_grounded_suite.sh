#!/usr/bin/env bash
# 从仓库根目录手动启动一组队列；失败即停，避免继续消耗后续实验预算。
set -euo pipefail

QUEUE="" BEAUTY_DATA="" SPORTS_DATA="" BEAUTY_SID="" SPORTS_SID="" BEAUTY_EMBED="" SPORTS_EMBED=""
SEED=42 NOTES="" DRY_RUN=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift; continue ;;
    --queue|--beauty-data-dir|--sports-data-dir|--beauty-semantic-id-path|--sports-semantic-id-path|--beauty-embedding-path|--sports-embedding-path|--seed|--notes)
      flag="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Error: $flag requires a value." >&2; exit 2; fi
      value="$2"; shift 2 ;;
    --queue=*|--beauty-data-dir=*|--sports-data-dir=*|--beauty-semantic-id-path=*|--sports-semantic-id-path=*|--beauty-embedding-path=*|--sports-embedding-path=*|--seed=*|--notes=*)
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
    --seed) SEED="$value" ;;
    --notes) NOTES="$value" ;;
  esac
done
for value in "$QUEUE" "$BEAUTY_DATA" "$SPORTS_DATA" "$BEAUTY_SID" "$SPORTS_SID" "$BEAUTY_EMBED" "$SPORTS_EMBED"; do
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: queue and both datasets' data-dir, semantic-id-path and embedding-path are required." >&2; exit 2; }
done
[[ "$QUEUE" == 1 || "$QUEUE" == 2 ]] || { echo "Error: queue must be 1 or 2." >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Error: seed must be a nonnegative integer." >&2; exit 2; }
if [[ "$QUEUE" == 1 ]]; then
  GPU_PAIR=0,1 PORT=29710
  DATASETS=(beauty beauty beauty beauty sports sports)
  ARMS=(original full single_prototype shuffled original full)
else
  GPU_PAIR=2,3 PORT=29720
  DATASETS=(beauty beauty beauty beauty sports sports)
  ARMS=(mask_ce hybrid token_content_init no_aux mask_ce hybrid)
fi
for index in "${!ARMS[@]}"; do
  dataset="${DATASETS[$index]}"; arm="${ARMS[$index]}"
  if [[ "$dataset" == beauty ]]; then
    data="$BEAUTY_DATA"; sid="$BEAUTY_SID"; embedding="$BEAUTY_EMBED"
  else
    data="$SPORTS_DATA"; sid="$SPORTS_SID"; embedding="$SPORTS_EMBED"
  fi
  ARGS=(--data-dir "$data" --semantic-id-path "$sid" --embedding-path "$embedding"
    --dataset "$dataset" --group rkmeans --devices '[0,1]' --arm "$arm" --seed "$SEED" --master-port "$PORT"
    --notes "CGBS first matrix; dataset=$dataset; arm=$arm; seed=$SEED; queue=$QUEUE; from scratch; ${NOTES}")
  if [[ "$DRY_RUN" == true ]]; then ARGS+=(--dry-run); fi
  ARGS+=("${EXTRA_ARGS[@]}")
  echo "CGBS queue=$QUEUE experiment=$((index + 1))/6 dataset=$dataset arm=$arm GPUs=$GPU_PAIR"
  CUDA_VISIBLE_DEVICES="$GPU_PAIR" NPROC_PER_NODE=2 bash ./tiger_catalog_grounded_train.sh "${ARGS[@]}"
done
