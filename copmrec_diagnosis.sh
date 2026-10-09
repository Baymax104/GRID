#!/usr/bin/env bash
set -euo pipefail
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
source ./liger_common.sh
source ./copmrec_m3_common.sh
[[ "${NPROC_PER_NODE:-1}" == 1 ]] || { echo "Error: M3 diagnosis requires one GPU process." >&2; exit 2; }
copmrec_m3_parse "$@"
m3_variant="${m3_variant:-full}"
case "$m3_analysis/$m3_variant" in
  hits/full) [[ -z "$m3_sha" && -z "$m3_checkpoint" ]] || { echo "Error: hits uses Testing bundles, no checkpoint." >&2; exit 2; } ;;
  residual/full|prefix/full|prefix/no_mixture)
    copmrec_m3_require_sha
    [[ -n "$m3_checkpoint" && "$m3_checkpoint" != null ]] || { echo "Error: checkpoint diagnosis requires --checkpoint." >&2; exit 2; } ;;
  *) echo "Error: use hits/full, residual/full, or prefix/{full,no_mixture}." >&2; exit 2 ;;
esac
# train 模式只复用公共参数解析；experiment.run_mode=analysis 调用 Trainer.test。
identity_args=()
if [[ "$m3_analysis" != hits ]]; then
  identity_args+=("checkpoint_sha256=\"$m3_sha\"")
fi
liger_launch train experiment=copmrec_diagnosis "diagnosis_analysis=$m3_analysis" "diagnosis_variant=$m3_variant" "${identity_args[@]}" "${m3_args[@]}"
