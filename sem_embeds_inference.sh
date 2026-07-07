#!/bin/bash

NPROC_PER_NODE=1

OMP_NUM_THREADS=$(( $(nproc) / NPROC_PER_NODE )) \
  uv run \
  torchrun --nproc_per_node=$NPROC_PER_NODE -m src.main \
  experiment=sem_embeds_inference \
  devices=[0] \
  data_dir=data/beauty --dry-run
