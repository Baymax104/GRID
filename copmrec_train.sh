#!/usr/bin/env bash
set -euo pipefail
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
source ./liger_common.sh
[[ "$NPROC_PER_NODE" == 2 ]] || { echo "Error: CoPMRec formal training requires two GPU processes." >&2; exit 2; }

common_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|--checkpoint=*|ckpt_path=*|+ckpt_path=*|++ckpt_path=*)
      [[ "$1" == ckpt_path=null ]] || { echo "Error: fresh formal training requires ckpt_path=null." >&2; exit 2; }
      common_args+=("$1")
      shift
      ;;
    --data-dir|--semantic-id-path|--embedding-path|--dataset|--devices|--notes|--group|--seed|--master-port)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a non-empty value." >&2; exit 2; }
      common_args+=("$1" "$2")
      shift 2
      ;;
    *)
      [[ -n "${1//[[:space:]]/}" ]] || { echo "Error: empty Hydra override." >&2; exit 2; }
      common_args+=("$1")
      shift
      ;;
  esac
done

# 正式训练从头开始，开发阶段 checkpoint 不具有正式实验资格。
liger_launch train experiment=copmrec_train ckpt_path=null "${common_args[@]}"
