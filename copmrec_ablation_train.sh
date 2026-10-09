#!/usr/bin/env bash
set -euo pipefail
source ./copmrec_m3_common.sh
copmrec_m3_parse "$@"
copmrec_m3_require_variant
[[ -z "$m3_analysis" && -z "$m3_sha" ]] || { echo "Error: training does not accept diagnosis/checkpoint identity." >&2; exit 2; }
bash ./copmrec_train.sh experiment=copmrec_ablation_train "ablation_variant=$m3_variant" "${m3_args[@]}"
