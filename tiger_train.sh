#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=tiger_train \
  devices=[0,1] \
  semantic_id_path=logs/rkmeans_inference/runs/2026-07-09/15-40-40/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  num_hierarchies=4 --dry-run
