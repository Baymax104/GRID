#!/usr/bin/env bash
set -euo pipefail
source ./copmrec_m3_common.sh
copmrec_m3_parse "$@"
copmrec_m3_require_variant
copmrec_m3_require_sha
[[ -z "$m3_analysis" ]] || { echo "Error: inference does not accept --analysis." >&2; exit 2; }
bash ./copmrec_inference.sh experiment=copmrec_ablation_inference "ablation_variant=$m3_variant" --checkpoint-sha256 "$m3_sha" "${m3_args[@]}"
