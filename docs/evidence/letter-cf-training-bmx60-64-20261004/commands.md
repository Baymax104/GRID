# BMX-60～64 CF 导出命令（2026-10-04）

CF训练核验后固定引用各自evaluation NDCG@10最大checkpoint。由用户从node1仓库根目录手动执行。以下命令未启动；GPU0为物理GPU，单进程Trainer使用逻辑devices=[0]。来源和恢复限制详见同目录audit.json及Linear当前正式结果。

## BMX-60 / beauty / seed42

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/5utubi3y?role=checkpoint&alias=v0&file=step_step=044000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29560
bash ./letter_cf_export.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_cf_export; dataset=beauty; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

## BMX-61 / beauty / seed200

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/ize3pkzo?role=checkpoint&alias=v2&file=step_step=024000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29561
bash ./letter_cf_export.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_cf_export; dataset=beauty; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

## BMX-62 / beauty / seed2026

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/yhm7t0gv?role=checkpoint&alias=v1&file=step_step=042000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29562
bash ./letter_cf_export.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_cf_export; dataset=beauty; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

## BMX-63 / sports / seed42

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/pvinh0ml?role=checkpoint&alias=v0&file=step_step=033000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29563
bash ./letter_cf_export.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_cf_export; dataset=sports; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

## BMX-64 / sports / seed200

```bash
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/h3bhzool?role=checkpoint&alias=v1&file=step_step=014000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29564
bash ./letter_cf_export.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_cf_export; dataset=sports; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```
