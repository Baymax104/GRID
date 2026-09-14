#!/usr/bin/env bash
# 仅复用显式 split outcome diagnosis；MIR trace 不是旧 prefix survival。
set -euo pipefail
bash ./tiger_catalog_grounded_diagnosis.sh "$@" experiment=tiger_item_resolution_diagnosis
