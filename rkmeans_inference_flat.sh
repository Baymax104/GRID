#!/bin/bash

NPROC_PER_NODE=2

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.inference \
  experiment=rkmeans_inference_flat \
  embedding_path=logs/sem_embeds_inference_flat/runs/2026-04-19/22-35-00/pickle/merged_predictions_tensor.pt \
  ckpt_path=logs/rkmeans_train_flat/runs/2026-04-19/23-07-35/checkpoints/checkpoint_000_000300.ckpt \
  data_dir=data/beauty \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256
