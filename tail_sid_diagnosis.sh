#!/bin/bash
set -euo pipefail

SEMANTIC_ID_PATH=""
RECOMMENDATION_OUTPUT_PATH=""
WIDENED_RECOMMENDATION_OUTPUT_PATH=""
FIXED_PREFIX_TRACE_PATH=""
WIDENED_PREFIX_TRACE_PATH=""
BASELINE_RECOMMENDATION_OUTPUT_PATH=""
INTERVENTION_RECOMMENDATION_OUTPUT_PATH=""
BASELINE_PREFIX_TRACE_PATH=""
INTERVENTION_PREFIX_TRACE_PATH=""
CANDIDATE_ALLOCATION_PROBE=false
GROUP=""
NOTES=""
DATA_DIR=""
SEED="42"
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
    --data-dir=*)
      DATA_DIR="${1#--data-dir=}"
      shift
      ;;
    --data-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --data-dir requires a value." >&2
        exit 2
      fi
      DATA_DIR="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#--seed=}"
      shift
      ;;
    --seed)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --seed requires a value." >&2
        exit 2
      fi
      SEED="$2"
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
    --candidate-allocation-probe)
      CANDIDATE_ALLOCATION_PROBE=true
      shift
      ;;
    --widened-recommendation-output-path=*)
      WIDENED_RECOMMENDATION_OUTPUT_PATH="${1#--widened-recommendation-output-path=}"
      if [[ -z "$WIDENED_RECOMMENDATION_OUTPUT_PATH" ]]; then
        echo "Error: --widened-recommendation-output-path requires a non-empty value." >&2
        exit 2
      fi
      shift
      ;;
    --widened-recommendation-output-path)
      if [[ $# -lt 2 || "$2" == --* || -z "$2" ]]; then
        echo "Error: --widened-recommendation-output-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      WIDENED_RECOMMENDATION_OUTPUT_PATH="$2"
      shift 2
      ;;
    --fixed-prefix-trace-path=*)
      FIXED_PREFIX_TRACE_PATH="${1#--fixed-prefix-trace-path=}"
      shift
      ;;
    --fixed-prefix-trace-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --fixed-prefix-trace-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      FIXED_PREFIX_TRACE_PATH="$2"
      shift 2
      ;;
    --widened-prefix-trace-path=*)
      WIDENED_PREFIX_TRACE_PATH="${1#--widened-prefix-trace-path=}"
      shift
      ;;
    --widened-prefix-trace-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --widened-prefix-trace-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      WIDENED_PREFIX_TRACE_PATH="$2"
      shift 2
      ;;
    --baseline-recommendation-output-path=*)
      BASELINE_RECOMMENDATION_OUTPUT_PATH="${1#--baseline-recommendation-output-path=}"
      shift
      ;;
    --baseline-recommendation-output-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --baseline-recommendation-output-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      BASELINE_RECOMMENDATION_OUTPUT_PATH="$2"
      shift 2
      ;;
    --intervention-recommendation-output-path=*)
      INTERVENTION_RECOMMENDATION_OUTPUT_PATH="${1#--intervention-recommendation-output-path=}"
      shift
      ;;
    --intervention-recommendation-output-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --intervention-recommendation-output-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      INTERVENTION_RECOMMENDATION_OUTPUT_PATH="$2"
      shift 2
      ;;
    --baseline-prefix-trace-path=*)
      BASELINE_PREFIX_TRACE_PATH="${1#--baseline-prefix-trace-path=}"
      shift
      ;;
    --baseline-prefix-trace-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --baseline-prefix-trace-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      BASELINE_PREFIX_TRACE_PATH="$2"
      shift 2
      ;;
    --intervention-prefix-trace-path=*)
      INTERVENTION_PREFIX_TRACE_PATH="${1#--intervention-prefix-trace-path=}"
      shift
      ;;
    --intervention-prefix-trace-path)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Error: --intervention-prefix-trace-path requires a local path or wandb://<run-id> value." >&2
        exit 2
      fi
      INTERVENTION_PREFIX_TRACE_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Error: --data-dir requires a value." >&2
  exit 2
fi

if [[ -z "$SEED" ]]; then
  echo "Error: --seed requires a value." >&2
  exit 2
fi

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

if [[ "$CANDIDATE_ALLOCATION_PROBE" == true ]]; then
  for required_pair in \
    "$BASELINE_RECOMMENDATION_OUTPUT_PATH" \
    "$INTERVENTION_RECOMMENDATION_OUTPUT_PATH" \
    "$BASELINE_PREFIX_TRACE_PATH" \
    "$INTERVENTION_PREFIX_TRACE_PATH"; do
    if [[ -z "$required_pair" ]]; then
      echo "Error: --candidate-allocation-probe requires all four baseline/intervention paths." >&2
      exit 2
    fi
  done
fi

ARGS=(
  experiment=tail_sid_diagnosis
  group="$GROUP"
  data_dir="$DATA_DIR"
  seed="$SEED"
  raw_num_hierarchies=3
  semantic_id_path="$SEMANTIC_ID_PATH"
  embedding_path=wandb://01mw1fez
)

ARGS+=("logger.wandb.notes=$(quote_hydra_string "$NOTES")")

if [[ -n "$RECOMMENDATION_OUTPUT_PATH" ]]; then
  ARGS+=("recommendation_output_path=$RECOMMENDATION_OUTPUT_PATH")
fi

if [[ -n "$WIDENED_RECOMMENDATION_OUTPUT_PATH" ]]; then
  ARGS+=("widened_recommendation_output_path=$WIDENED_RECOMMENDATION_OUTPUT_PATH")
fi

if [[ -n "$FIXED_PREFIX_TRACE_PATH" ]]; then
  ARGS+=("fixed_prefix_trace_path=$FIXED_PREFIX_TRACE_PATH")
fi

if [[ -n "$WIDENED_PREFIX_TRACE_PATH" ]]; then
  ARGS+=("widened_prefix_trace_path=$WIDENED_PREFIX_TRACE_PATH")
fi

if [[ "$CANDIDATE_ALLOCATION_PROBE" == true ]]; then
  ARGS+=(
    candidate_allocation_probe.enabled=true
    "baseline_recommendation_output_path=$BASELINE_RECOMMENDATION_OUTPUT_PATH"
    "intervention_recommendation_output_path=$INTERVENTION_RECOMMENDATION_OUTPUT_PATH"
    "baseline_prefix_trace_path=$BASELINE_PREFIX_TRACE_PATH"
    "intervention_prefix_trace_path=$INTERVENTION_PREFIX_TRACE_PATH"
  )
fi

if [[ "$DRY_RUN" == true ]]; then
  ARGS+=(--dry-run)
fi

ARGS+=("${EXTRA_ARGS[@]}")

uv run --module src.main "${ARGS[@]}"
