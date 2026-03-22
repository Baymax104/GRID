#!/bin/bash

conda run -n GRID --no-capture-output \
  torchrun --nproc_per_node=2 -m src.inference \
  experiment=rkmeans_inference_flat \
  embedding_path=logs/inference/runs/2026-03-12/14-34-34/pickle/merged_predictions_tensor.pt \
  ckpt_path=logs/train/runs/2026-03-22/08-46-47/checkpoints/checkpoint_000_000300.ckpt \
  data_dir=data/beauty \
  embedding_dim=768 \
  num_hierarchies=3 \
  codebook_width=256
