#!/bin/bash

NPROC_PER_NODE=4

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=tiger_train \
  devices=[0,1,4,5] \
  semantic_id_path=logs/rkmeans_inference/runs/2026-04-21/16-33-50/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  num_hierarchies=4 \
