# BMX-60～65 Tokenizer训练命令

CF导出产物和checkpoint来源核验通过；每条命令可独立复制，从node1仓库根目录手动执行。物理GPU0映射Trainer逻辑devices=[0]。Tokenizer最多20000 epoch、batch1024、输入1024维内容和32维CF、latent32、4×256、10groups，每2000 epoch按全目录collision_rate最小选择best。保留已有冻结参数，不启动正式训练。CF训练历史来源/last恢复限制继续保留在各issue中。

## BMX-60 / beauty / seed42

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/c184aneu?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/5utubi3y?role=checkpoint&alias=v0&file=step_step=044000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29560
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_tokenizer_train; dataset=beauty; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

## BMX-61 / beauty / seed200

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/223be7uk?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/ize3pkzo?role=checkpoint&alias=v2&file=step_step=024000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29561
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_tokenizer_train; dataset=beauty; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

## BMX-62 / beauty / seed2026

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/5a5gmzoa?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/yhm7t0gv?role=checkpoint&alias=v1&file=step_step=042000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29562
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_tokenizer_train; dataset=beauty; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

## BMX-63 / sports / seed42

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/hsrnapw7?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/pvinh0ml?role=checkpoint&alias=v0&file=step_step=033000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29563
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_tokenizer_train; dataset=sports; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```

## BMX-64 / sports / seed200

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/6los4ob4?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/h3bhzool?role=checkpoint&alias=v1&file=step_step=014000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29564
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_tokenizer_train; dataset=sports; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```

## BMX-65 / sports / seed2026

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/fmxspnds?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/4kzf622t?role=checkpoint&alias=v2&file=step_step=018000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29565
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_tokenizer_train; dataset=sports; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```
