#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=rkmeans_inference \
  embedding_path=logs/sem_embeds_inference/runs/2026-07-06/23-50-25/pickle/merged_predictions_tensor.pt \
  ckpt_path=logs/rkmeans_train/runs/2026-07-09/15-37-00/checkpoints/checkpoint_000_001000.ckpt \
  devices=[0,1] \
  data_dir=data/beauty \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256
