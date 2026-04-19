#!/bin/bash

NPROC_PER_NODE=2

conda run -n GRID --no-capture-output \
  OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.train \
  +should_skip_retry=true \
  experiment=rkmeans_train_flat \
  embedding_path=logs/sem_embeds_inference_flat/runs/2026-03-22/23-38-29/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256
