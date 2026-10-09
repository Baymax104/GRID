#!/usr/bin/env bash
# SASRec 根目录入口共享参数解析。

sasrec_quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

sasrec_launch() {
  local mode="$1"
  shift
  local data_dir="" catalog_path="" dataset_name="" checkpoint="" notes=""
  local seed="42" devices="${DEVICES:-[0]}" group="${WANDB_GROUP:-sasrec}"
  local master_port="${MASTER_PORT:-29500}" nproc="${NPROC_PER_NODE:-}" dry_run=false
  local option value
  local -a extra_args=() device_ids=() command=() overrides=()
  if [[ ! -f pyproject.toml || ! -f src/main.py ]]; then
    echo "Error: run SASRec scripts from the GRID repository root." >&2
    return 2
  fi
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --dry-run ]]; then
      dry_run=true
      shift
      continue
    fi
    if [[ "$1" == --* ]]; then
      option="${1%%=*}"
      case "$option" in
        --data-dir|--item-catalog-path|--dataset-name|--checkpoint|--seed|--devices|--notes|--wandb-group|--master-port) ;;
        *) echo "Error: unknown SASRec option $option." >&2; return 2 ;;
      esac
      if [[ "$1" == *=* ]]; then
        value="${1#*=}"
        shift
      else
        if [[ $# -lt 2 || "$2" == --* ]]; then
          echo "Error: $option requires a value." >&2
          return 2
        fi
        value="$2"
        shift 2
      fi
      if [[ -z "$value" && "$option" != --notes ]]; then
        echo "Error: $option requires a nonempty value." >&2
        return 2
      fi
      case "$option" in
        --data-dir) data_dir="$value" ;;
        --item-catalog-path) catalog_path="$value" ;;
        --dataset-name) dataset_name="$value" ;;
        --checkpoint) checkpoint="$value" ;;
        --seed) seed="$value" ;;
        --devices) devices="$value" ;;
        --notes) notes="$value" ;;
        --wandb-group) group="$value" ;;
        --master-port) master_port="$value" ;;
      esac
    elif [[ "$1" == *=* && "$1" != -* ]]; then
      extra_args+=("$1")
      shift
    else
      echo "Error: expected a SASRec option or Hydra override, got $1." >&2
      return 2
    fi
  done
  if [[ -z "$data_dir" || -z "$catalog_path" || ! "$dataset_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: --data-dir, --item-catalog-path and --dataset-name are required." >&2
    return 2
  fi
  if [[ "$mode" == inference && -z "$checkpoint" ]]; then
    echo "Error: SASRec inference requires --checkpoint." >&2
    return 2
  fi
  if [[ ! "$seed" =~ ^[0-9]{1,10}$ ]] || (( 10#$seed < 1 || 10#$seed > 4294967295 )); then
    echo "Error: --seed must be an integer in 1..4294967295." >&2
    return 2
  fi
  if [[ ! "$master_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$master_port < 1 || 10#$master_port > 65535 )); then
    echo "Error: --master-port must be an integer in 1..65535." >&2
    return 2
  fi
  devices="${devices//[[:space:]]/}"
  if [[ ! "$devices" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
    echo "Error: --devices must be a nonempty GPU index list such as [0,1]." >&2
    return 2
  fi
  value="${devices#[}"
  value="${value%]}"
  IFS=',' read -r -a device_ids <<< "$value"
  nproc="${nproc:-${#device_ids[@]}}"
  if [[ ! "$nproc" =~ ^[1-9][0-9]*$ || "$nproc" != "${#device_ids[@]}" ]]; then
    echo "Error: NPROC_PER_NODE must equal the number of configured devices." >&2
    return 2
  fi
  if [[ "$nproc" == 1 ]]; then
    command=(uv run python -m src.main)
  else
    command=(uv run torchrun "--nproc_per_node=$nproc" "--master_port=$master_port" -m src.main)
  fi
  overrides=("experiment=sasrec_$mode" "data_dir=$(sasrec_quote_hydra_string "$data_dir")"
    "item_catalog_path=$(sasrec_quote_hydra_string "$catalog_path")" "dataset_name=$dataset_name"
    "devices=$devices" "seed=$seed" "group=$(sasrec_quote_hydra_string "$group")" "dry_run=$dry_run")
  if [[ -n "$checkpoint" ]]; then
    overrides+=("ckpt_path=$(sasrec_quote_hydra_string "$checkpoint")")
  fi
  if [[ -n "$notes" ]]; then
    overrides+=("logger.wandb.notes=$(sasrec_quote_hydra_string "$notes")")
  fi
  "${command[@]}" "${overrides[@]}" "${extra_args[@]}"
}
