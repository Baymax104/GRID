#!/usr/bin/env bash
set -euo pipefail
source ./sasrec_common.sh
sasrec_launch train "$@"
