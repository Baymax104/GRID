#!/usr/bin/env bash
# LETTER独立入口，公用框架通过src.main装配。
set -euo pipefail

letter_main() {
  local stage="$1"; shift
  local dataset="" data_dir="" embedding="" cf="" cf_source="" sid="" ckpt="" notes="" seed=42
  local gpus="0" nproc=1 port=29521 dry=false
  local -a extra=() defaults=() command=()
  if [[ "$stage" == letter_train || "$stage" == letter_inference ]]; then gpus="0,1"; nproc=2; fi
  while (( $# )); do
    case "$1" in
      --dry-run) dry=true; shift ;;
      --dataset|--data-dir|--embedding-path|--cf-embedding-path|--cf-source|--semantic-id-path|--ckpt-path|--notes|--seed|--gpus|--nproc-per-node|--master-port)
        if (( $# < 2 )) || [[ "$2" == --* ]]; then echo "Missing value: $1" >&2; return 2; fi
        case "$1" in
          --dataset) dataset="$2" ;; --data-dir) data_dir="$2" ;;
          --embedding-path) embedding="$2" ;; --cf-embedding-path) cf="$2" ;;
          --cf-source) cf_source="$2" ;; --semantic-id-path) sid="$2" ;;
          --ckpt-path) ckpt="$2" ;; --notes) notes="$2" ;; --seed) seed="$2" ;;
          --gpus) gpus="$2" ;; --nproc-per-node) nproc="$2" ;; --master-port) port="$2" ;;
        esac
        shift 2 ;;
      --*=*)
        local key="${1%%=*}" value="${1#*=}"
        # 统一两个参数形态，保留空notes；其他空值在必填验证中拒绝。
        set -- "$key" "$value" "${@:2}" ;;
      --*) echo "Unknown option: $1" >&2; return 2 ;;
      *) [[ "$1" == *=* && "$1" != =* ]] || { echo "Invalid Hydra override: $1" >&2; return 2; }; extra+=("$1"); shift ;;
    esac
  done
  [[ "$dataset" =~ ^(beauty|sports|toys)$ ]] || { echo 'dataset must be beauty/sports/toys' >&2; return 2; }
  [[ "$seed" =~ ^[0-9]+$ && "$nproc" =~ ^[1-9][0-9]*$ && "$port" =~ ^[0-9]+$ ]] || { echo 'Invalid numeric option' >&2; return 2; }
  [[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU list' >&2; return 2; }
  local -a gpu_array=(); IFS=',' read -r -a gpu_array <<< "$gpus"
  (( nproc <= ${#gpu_array[@]} && 10#$port > 0 && 10#$port < 65536 )) || { echo 'Invalid GPU count or port' >&2; return 2; }
  if [[ "$stage" == letter_cf_train || "$stage" == letter_cf_export ]]; then
    [[ -n "$embedding" && -n "$data_dir" && "$nproc" == 1 ]] || { echo 'CF requires embedding, data-dir and one process' >&2; return 2; }
    defaults+=("embedding_path=$(letter_quote "$embedding")" "data_dir=$(letter_quote "$data_dir")")
  elif [[ "$stage" == letter_tokenizer_train || "$stage" == letter_sid ]]; then
    [[ -n "$embedding" && -n "$cf" && -n "$cf_source" && "$nproc" == 1 ]] || { echo 'Tokenizer requires embedding, CF, CF source and one process' >&2; return 2; }
    defaults+=("embedding_path=$(letter_quote "$embedding")" "cf_embedding_path=$(letter_quote "$cf")" "cf_source=$(letter_quote "$cf_source")")
  else
    [[ -n "$data_dir" && -n "$sid" ]] || { echo 'Recommendation requires data-dir and semantic-id-path' >&2; return 2; }
    defaults+=("data_dir=$(letter_quote "$data_dir")" "semantic_id_path=$(letter_quote "$sid")")
  fi
  if [[ "$stage" == letter_sid || "$stage" == letter_inference || "$stage" == letter_cf_export ]]; then
    [[ -n "$ckpt" ]] || { echo 'Inference requires an explicit checkpoint' >&2; return 2; }
    defaults+=("ckpt_path=$(letter_quote "$ckpt")")
  elif [[ -n "$ckpt" ]]; then defaults+=("ckpt_path=$(letter_quote "$ckpt")"); fi
  local devices='[' i
  for (( i=0; i<nproc; i++ )); do [[ $i == 0 ]] || devices+=','; devices+="$i"; done
  devices+=']'
  defaults+=("dataset_name=$dataset" "seed=$seed" "devices=$devices" "dry_run=$dry")
  [[ -z "$notes" ]] || defaults+=("logger.wandb.notes=$(letter_quote "$notes")")
  if (( nproc > 1 )); then command=(uv run torchrun "--nproc_per_node=$nproc" "--master_port=$port" -m src.main)
  else command=(uv run python -m src.main); fi
  export CUDA_VISIBLE_DEVICES="$gpus"
  "${command[@]}" "experiment=$stage" "${defaults[@]}" "${extra[@]}"
}

letter_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
