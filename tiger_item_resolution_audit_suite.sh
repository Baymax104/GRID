#!/usr/bin/env bash
# 四个已完成 checkpoint 串行取证，不启动训练。
set -euo pipefail
BEAUTY_DATA="" SPORTS_DATA="" GPU=0 NOTES="" EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --beauty-data-dir|--sports-data-dir|--gpu|--notes)
      flag="$1"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Error: $flag requires a value." >&2; exit 2; }
      value="$2"; shift 2 ;;
    --beauty-data-dir=*|--sports-data-dir=*|--gpu=*|--notes=*)
      flag="${1%%=*}"; value="${1#*=}"; shift ;;
    --dry-run|--print-only) EXTRA_ARGS+=("$1"); shift; continue ;;
    --*) echo "Error: unknown option $1" >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift; continue ;;
  esac
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: $flag requires a nonempty value." >&2; exit 2; }
  case "$flag" in
    --beauty-data-dir) BEAUTY_DATA="$value" ;;
    --sports-data-dir) SPORTS_DATA="$value" ;;
    --gpu) GPU="$value" ;;
    --notes) NOTES="$value" ;;
  esac
done
[[ -n "$BEAUTY_DATA" && -n "$SPORTS_DATA" && "$GPU" =~ ^[0-9]+$ ]] || { echo "Error: both data directories and one GPU are required." >&2; exit 2; }
for dataset in beauty sports; do
  if [[ "$dataset" == beauty ]]; then
    DATA="$BEAUTY_DATA" SID=wandb://4vyi4o6w EMBED=wandb://3jtt9mpa MIR=wgx7944n FIXED=m47u9t4k
  else
    DATA="$SPORTS_DATA" SID=wandb://3narllqy EMBED=wandb://psec3u5i MIR=y2ilymxo FIXED=6jt7s9ro
  fi
  for arm in mir depth2; do
    if [[ "$arm" == mir ]]; then CHECKPOINT="$MIR"; else CHECKPOINT="$FIXED"; fi
    bash ./tiger_item_resolution_audit.sh \
      --data-dir "$DATA" --dataset "$dataset" --semantic-id-path "$SID" --embedding-path "$EMBED" \
      --checkpoint-path "wandb://$CHECKPOINT" --arm "$arm" --seed 42 --gpu "$GPU" --data-split evaluation \
      --notes "MIR score/search decision audit; frozen best checkpoint=$CHECKPOINT; paired hash sample; no training; $NOTES" \
      "${EXTRA_ARGS[@]}"
  done
done
