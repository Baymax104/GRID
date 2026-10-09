# BMX-65 CF导出（Sports / seed2026）

验证集NDCG@10选出的best为step18000；由用户从node1仓库根目录手动执行。物理GPU0映射Trainer逻辑devices=[0]。last.ckpt停在18000，不是50k恢复点。

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/4kzf622t?role=checkpoint&alias=v2&file=step_step=018000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29565
bash ./letter_cf_export.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_cf_export; dataset=sports; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```
