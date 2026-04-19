#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.train \
  experiment=tiger_train_flat \
  +should_skip_retry=true \
  semantic_id_path=logs/rkmeans_inference_flat/runs/2026-04-19/23-11-11/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  num_hierarchies=4 \

