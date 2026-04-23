#!/bin/bash

NPROC_PER_NODE=4

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.train \
  +should_skip_retry=true \
  experiment=rkmeans_train_flat \
  embedding_path=logs/sem_embeds_inference_flat/runs/2026-04-19/22-35-00/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  devices=[0,1,2,3] \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256
