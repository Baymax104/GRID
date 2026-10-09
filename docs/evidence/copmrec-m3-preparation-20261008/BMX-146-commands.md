## 6. 执行步骤与命令

入口已实现并完成本地验证，正式运行未启动。三个来源各执行一次单卡 Trainer.test；每个块都可从 GRID 根目录独立复制，物理 GPU 0 → CUDA local [0]。Full 精确来源已填；A1/A4 必须等其自身训练选点审计后补齐 URI/SHA，不能从 Full 初始化或借用 Full checkpoint。

### 来源 full（BMX-120）

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt'
export COPMREC_CHECKPOINT_SHA256='a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4'

bash ./copmrec_diagnosis.sh \
  --analysis prefix --variant full \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=full; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations'
```

### 来源 no_mixture（BMX-122）

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='【待填写：BMX-122 自身训练的 validation-selected best 不可变 URI】'
export COPMREC_CHECKPOINT_SHA256='【待填写：该 best 文件的 SHA256】'

bash ./copmrec_diagnosis.sh \
  --analysis prefix --variant no_mixture \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=no_mixture; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations'
```

### 来源 legal_generation（BMX-142）

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='【待填写：BMX-142 自身训练的 validation-selected best 不可变 URI】'
export COPMREC_CHECKPOINT_SHA256='【待填写：该 best 文件的 SHA256】'

bash ./copmrec_diagnosis.sh \
  --analysis prefix --variant legal_generation \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=legal_generation; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations'
```

按真实目标前缀测量逐层 gen/mass 的概率、NLL、目标rank、熵、JS 与 argmax 一致率；Full 另测 mixed 及全局 alpha，A1/A4 的 mixed/alpha 字段 null。预定义概率差分箱、训练频次/seen/历史切片，保留空组和实际N。共享 writer 发布最终观测，不把条件化概率解释成 free-running 候选恢复。诊断仅 eval，无优化器更新、重新选点或部署改动。
