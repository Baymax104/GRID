#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=tiger_train \
  devices=[0,1] \
  codebook_size=256 \
  semantic_id_path=logs/rkmeans_inference/runs/2026-08-06/15-09-18/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  num_hierarchies=3 --dry-run
