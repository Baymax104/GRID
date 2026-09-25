#!/usr/bin/env bash
set -euo pipefail
source scripts/liger_common.sh
liger_launch inference experiment=liger_joint_inference --group liger_joint_mixture_v1 "$@"
