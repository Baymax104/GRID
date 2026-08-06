#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=rkmeans_train \
  embedding_path=logs/sem_embeds_inference/runs/2026-08-06/11-30-14/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  devices=[0,1] \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_size=256 --dry-run
