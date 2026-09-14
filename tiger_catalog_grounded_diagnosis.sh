#!/usr/bin/env bash
# 复用已有 diagnosis 参数与统一入口，仅显式补充数据划分和内容来源。
set -euo pipefail
SPLIT="" EMBEDDING=""
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-split|--embedding-path)
      flag="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then echo "Error: $flag requires a non-empty value." >&2; exit 2; fi
      value="$2"; shift 2 ;;
    --data-split=*|--embedding-path=*) flag="${1%%=*}"; value="${1#*=}"; shift ;;
    *) FORWARDED+=("$1"); shift; continue ;;
  esac
  [[ -n "${value//[[:space:]]/}" ]] || { echo "Error: $flag requires a non-empty value." >&2; exit 2; }
  case "$flag" in --data-split) SPLIT="$value" ;; --embedding-path) EMBEDDING="$value" ;; esac
done
[[ "$SPLIT" == evaluation || "$SPLIT" == testing ]] || { echo "Error: data-split must be evaluation or testing." >&2; exit 2; }
[[ -n "$EMBEDDING" ]] || { echo "Error: embedding-path is required." >&2; exit 2; }
EMBEDDING="${EMBEDDING//\\/\\\\}"
EMBEDDING="${EMBEDDING//\"/\\\"}"
bash ./tail_sid_diagnosis.sh experiment=tiger_catalog_grounded_diagnosis "data_split=$SPLIT" \
  "embedding_path=\"$EMBEDDING\"" "${FORWARDED[@]}"
