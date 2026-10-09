# LETTER Toys CF 导出（2026-10-09）

用户将上次请求从 CF teacher test 纠正为 CF 导出。已撤回新增 CF test 脚本、配置、模型 test_step、analysis checkpoint 改动、相应测试与 OpenSpec 文件；issue 的 CF test 命令和结果段已撤回。随后按用户要求删除 W&B test run `pvau5wcu`、`1ucqm40x`、`rizj8teo`，已回读确认目标 run 无残留；artifact 和本地/node1 logs 保留，不作为后续输入。清理记录见 `logs/letter-toys-cf-test-20261009/wandb-cleanup.json`。本次回退通过28项聚焦测试、Ruff与现有配置 compose；Mutagen三session Watching for changes、无 conflict。

## 导出结果

三组均 finished、退出码0，完整目录 11,924 件商品，bundle 为 keys/predictions，shape=[11924,32]。key唯一且与内容目录一致，向量有限，逐元素等于各自 best checkpoint 去掉 padding 后的未归一化商品表。发布 bundle digest、固定 checkpoint/content lineage 与 source verified 来源均通过。

| Issue | Seed | 物理 GPU | 导出 run | Artifact |
| --- | ---: | ---: | --- | --- |
| BMX-66 | 42 | 2 → CUDA0 | [jx7hith4](https://wandb.ai/baymaxam/GRID/runs/jx7hith4) | letter_cf_export_toys-collaborative-embedding:v2 |
| BMX-67 | 200 | 4 → CUDA0 | [ljowhf8y](https://wandb.ai/baymaxam/GRID/runs/ljowhf8y) | letter_cf_export_toys-collaborative-embedding:v1 |
| BMX-68 | 2026 | 5 → CUDA0 | [dwkqg7ud](https://wandb.ai/baymaxam/GRID/runs/dwkqg7ud) | letter_cf_export_toys-collaborative-embedding:v0 |

## 固定命令与下游输入

从 node1 仓库根目录 `/data3/weizhenyu/projects/GRID` 执行。以下命令仅导出，完整实际启动命令（含唯一 run id）保存在 export-plan.json。

### BMX-66 / seed42

```bash
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path 'wandb://baymaxam/GRID/wsflubzp?role=checkpoint&alias=v2&file=step_step=039000.ckpt' \
  --dataset toys \
  --seed 42 \
  --gpus 2 \
  --nproc-per-node 1 \
  --master-port 30166 \
  --notes 'issue=BMX-66; LETTER Toys; stage=letter_cf_export; seed=42; fixed evaluation-selected best' \
  logger.wandb.group=paper_main_letter_toys
```

已核验下游输入：

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/jx7hith4?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/wsflubzp?role=checkpoint&alias=v2&file=step_step=039000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
```

### BMX-67 / seed200

```bash
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path 'wandb://baymaxam/GRID/yix316dz?role=checkpoint&alias=v0&file=step_step=044000.ckpt' \
  --dataset toys \
  --seed 200 \
  --gpus 4 \
  --nproc-per-node 1 \
  --master-port 30167 \
  --notes 'issue=BMX-67; LETTER Toys; stage=letter_cf_export; seed=200; fixed evaluation-selected best' \
  logger.wandb.group=paper_main_letter_toys
```

已核验下游输入：

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/ljowhf8y?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/yix316dz?role=checkpoint&alias=v0&file=step_step=044000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
```

### BMX-68 / seed2026

```bash
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path 'wandb://baymaxam/GRID/q7cblret?role=checkpoint&alias=v1&file=step_step=043000.ckpt' \
  --dataset toys \
  --seed 2026 \
  --gpus 5 \
  --nproc-per-node 1 \
  --master-port 30168 \
  --notes 'issue=BMX-68; LETTER Toys; stage=letter_cf_export; seed=2026; fixed evaluation-selected best' \
  logger.wandb.group=paper_main_letter_toys
```

已核验下游输入：

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/dwkqg7ud?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/q7cblret?role=checkpoint&alias=v1&file=step_step=043000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
```

## 证据与阶段边界

本地与 node1 的 `logs/letter-toys-cf-export-20261009/export-plan.json` 和 `export-verification.json` 保存命令、GPU/tmux、固定输入、文件哈希和逐组核验。node1同目录保留脚本、日志、退出码。CF训练核验保存在 `logs/letter-toys-20261009-e94ie8w9/cf-completion-verification.json`。

本次仅完成 CF 导出，未启动 Tokenizer、SID 或推荐训练。BMX-66/67/68 保持 In Progress。

