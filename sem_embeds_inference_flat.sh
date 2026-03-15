#!/bin/bash

conda run -n GRID --no-capture-output \
  torchrun --nproc_per_node=2 -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir=data/beauty