#!/usr/bin/env bash
set -euo pipefail
source scripts/liger_common.sh
liger_launch train "$@"
