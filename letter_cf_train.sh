#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/letter_common.sh"
letter_main letter_cf_train "$@"
