#!/usr/bin/env bash
set -euo pipefail
source ./liger_common.sh
liger_launch train experiment=liger_joint_train --group liger_joint_mixture_v1 "$@"
