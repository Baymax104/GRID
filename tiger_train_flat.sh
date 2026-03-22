#!/bin/bash

conda run -n GRID --no-capture-output \
  torchrun --nproc_per_node=2 -m src.train \
  experiment=tiger_train_flat \
  semantic_id_path=logs/inference/runs/2026-03-22/09-03-45/pickle/merged_predictions_tensor.pt \
  data_dir=data/beauty \
  num_hierarchies=4 \

