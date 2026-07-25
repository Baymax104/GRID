#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=rvq_train \
  embedding_path=logs/sem_embeds_inference/runs/2026-07-24/00-32-49/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  devices=[0,1] \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256 --dry-run
