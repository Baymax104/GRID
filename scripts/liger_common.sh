#!/usr/bin/env bash
# Shared argument parser for the root LIGER train and inference launchers.
set -euo pipefail

[[ -f src/main.py ]] || { echo "Error: run from the GRID repository root." >&2; exit 2; }

liger_quote_hydra_string() {
  local value="$1"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

liger_launch() {
  local mode="${1:-}"
  [[ "$mode" == train || "$mode" == inference ]] || {
    echo "Error: mode must be train or inference." >&2
    return 2
  }
  shift

  local data_dir="" semantic_id="" embedding="" checkpoint="" dataset=""
  local notes="" devices="" group="" seed=""
  local dry_run=false flag value
  local master_port="${MASTER_PORT:-29730}"
  local nproc_per_node="${NPROC_PER_NODE:-1}"
  local -a extra_args=() args=() command=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dry-run)
        dry_run=true
        shift
        continue
        ;;
      --data-dir|--semantic-id-path|--embedding-path|--checkpoint|--dataset|--devices|--notes|--group|--seed)
        flag="$1"
        [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
          echo "Error: $flag requires a non-empty value." >&2
          return 2
        }
        value="$2"
        shift 2
        ;;
      --data-dir=*|--semantic-id-path=*|--embedding-path=*|--checkpoint=*|--dataset=*|--devices=*|--notes=*|--group=*|--seed=*)
        flag="${1%%=*}"
        value="${1#*=}"
        shift
        ;;
      --*)
        echo "Error: unknown option $1" >&2
        return 2
        ;;
      *)
        extra_args+=("$1")
        shift
        continue
        ;;
    esac

    [[ -n "${value//[[:space:]]/}" ]] || {
      echo "Error: $flag requires a non-empty value." >&2
      return 2
    }
    case "$flag" in
      --data-dir) data_dir="$value" ;;
      --semantic-id-path) semantic_id="$value" ;;
      --embedding-path) embedding="$value" ;;
      --checkpoint) checkpoint="$value" ;;
      --dataset) dataset="$value" ;;
      --devices) devices="$value" ;;
      --notes) notes="$value" ;;
      --group) group="$value" ;;
      --seed) seed="$value" ;;
    esac
  done

  for value in "$data_dir" "$semantic_id" "$embedding" "$dataset" "$devices"; do
    [[ -n "$value" ]] || {
      echo "Error: data-dir, dataset, semantic-id-path, embedding-path and devices are required." >&2
      return 2
    }
  done
  if [[ "$mode" == inference && -z "$checkpoint" ]]; then
    echo "Error: checkpoint is required for inference." >&2
    return 2
  fi
  [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: invalid process count." >&2
    return 2
  }
  [[ "$master_port" =~ ^[0-9]{1,5}$ ]] &&
    (( 10#$master_port > 0 && 10#$master_port <= 65535 )) || {
      echo "Error: invalid master port." >&2
      return 2
    }
  if [[ -n "$seed" && ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid seed." >&2
    return 2
  fi

  args=(
    "experiment=liger_${mode}"
    "data_dir=$(liger_quote_hydra_string "$data_dir")"
    "semantic_id_path=$(liger_quote_hydra_string "$semantic_id")"
    "embedding_path=$(liger_quote_hydra_string "$embedding")"
    "dataset_name=$(liger_quote_hydra_string "$dataset")"
    "devices=$devices"
  )
  [[ -z "$checkpoint" ]] || args+=("ckpt_path=$(liger_quote_hydra_string "$checkpoint")")
  [[ -z "$notes" ]] || args+=("logger.wandb.notes=$(liger_quote_hydra_string "$notes")")
  [[ -z "$group" ]] || args+=("group=$(liger_quote_hydra_string "$group")")
  [[ -z "$seed" ]] || args+=("seed=$seed")
  [[ "$dry_run" == false ]] || args+=(--dry-run)
  args+=("${extra_args[@]}")

  if (( 10#$nproc_per_node > 1 )); then
    command=(uv run torchrun "--nproc_per_node=$nproc_per_node" "--master_port=$master_port" -m src.main)
  else
    command=(uv run -m src.main)
  fi
  "${command[@]}" "${args[@]}"
}
