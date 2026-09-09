#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICES="${DEVICES:-[0,1]}"

SEMANTIC_ID_PATH=""
GROUP=""
NOTES=""
DRY_RUN=false
EXTRA_ARGS=()

quote_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --master-port=*)
      MASTER_PORT="${1#--master-port=}"
      shift
      ;;
    --master-port)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --master-port requires a value." >&2
        exit 2
      fi
      MASTER_PORT="$2"
      shift 2
      ;;
    --devices=*)
      DEVICES="${1#--devices=}"
      shift
      ;;
    --devices)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --devices requires a value." >&2
        exit 2
      fi
      DEVICES="$2"
      shift 2
      ;;
    --notes=*)
      NOTES="${1#--notes=}"
      shift
      ;;
    --notes)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --notes requires a value." >&2
        exit 2
      fi
      NOTES="$2"
      shift 2
      ;;
    --group=*)
      GROUP="${1#--group=}"
      shift
      ;;
    --group)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --group requires one of: rkmeans, rvq, rqvae." >&2
        exit 2
      fi
      GROUP="$2"
      shift 2
      ;;
    --semantic-id-path=*)
      SEMANTIC_ID_PATH="${1#--semantic-id-path=}"
      shift
      ;;
    --semantic-id-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --semantic-id-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      SEMANTIC_ID_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$NOTES" ]]; then
  echo "Error: --notes requires a value." >&2
  exit 2
fi

if [[ -z "$SEMANTIC_ID_PATH" ]]; then
  echo "Error: --semantic-id-path requires a local path or wandb://<run-id> value." >&2
  exit 2
fi

case "$GROUP" in
  rkmeans|rvq|rqvae)
    ;;
  "")
    echo "Error: --group requires one of: rkmeans, rvq, rqvae." >&2
    exit 2
    ;;
  *)
    echo "Error: unsupported --group '$GROUP'; expected one of: rkmeans, rvq, rqvae." >&2
    exit 2
    ;;
esac

ARGS=(
  experiment=tiger_train
  group="$GROUP"
  devices="$DEVICES"
  semantic_id_path="$SEMANTIC_ID_PATH"
  data_dir=data/sports
)

ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE")
if [[ -n "$MASTER_PORT" ]]; then
  TORCHRUN_ARGS+=(--master_port="$MASTER_PORT")
fi

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun "${TORCHRUN_ARGS[@]}" -m src.main \
  "${ARGS[@]}"
