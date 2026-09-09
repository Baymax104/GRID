#!/bin/bash
set -euo pipefail

SEMANTIC_ID_PATH=""
RECOMMENDATION_OUTPUT_PATH=""
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
    --recommendation-output-path=*)
      RECOMMENDATION_OUTPUT_PATH="${1#--recommendation-output-path=}"
      shift
      ;;
    --recommendation-output-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --recommendation-output-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      RECOMMENDATION_OUTPUT_PATH="$2"
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
  experiment=tail_sid_diagnosis
  group="$GROUP"
  data_dir=data/beauty
  raw_num_hierarchies=3
  semantic_id_path="$SEMANTIC_ID_PATH"
  embedding_path=wandb://01mw1fez
)

ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")

if [[ -n "$RECOMMENDATION_OUTPUT_PATH" ]]; then
  ARGS+=("recommendation_output_path=$RECOMMENDATION_OUTPUT_PATH")
fi

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

uv run --module src.main "${ARGS[@]}"
