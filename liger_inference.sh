#!/usr/bin/env bash
set -euo pipefail
source ./liger_common.sh
liger_launch inference "$@"
