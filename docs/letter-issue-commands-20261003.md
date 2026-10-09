# LETTER 九槽位命令交付

BMX-58可运行准备交付；九个正式实验槽位待用户手动启动。本文件与Linear子issue保持同一模板和命令，不包含未来尚未生成的虚构Artifact。

2026-10-04更新：BMX-60～64的CF训练已完成并核验，CF导出命令已固定各自best版本。下方原始准备状态为2026-10-03快照；当前来源和恢复限制见 [CF训练核验](evidence/letter-cf-training-bmx60-64-20261004/audit.json) 和各Linear issue，后续阶段仍待手动执行。

2026-10-04后续更新：BMX-60～65的CF导出已完成并核验，Tokenizer训练命令已补齐各自CF bundle和checkpoint。可独立复制的当前命令见 [Tokenizer训练命令](evidence/letter-tokenizer-inputs-bmx60-65-20261004/commands.md)；下方2026-10-03准备状态保留为历史快照，当前阶段以各Linear issue最新核验为准。

2026-10-06更新：BMX-60～65 Tokenizer训练已核验，SID导出命令已固定各自val/collision_rate选出的best，并补齐全部CF输入变量。当前可复制命令见 [SID导出命令](evidence/letter-tokenizer-training-bmx60-65-20261006/commands.md)；历史准备状态及既有训练来源限制保留，后续SID与推荐/Testing仍由用户手动执行。

# BMX-60

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Beauty
* Seed：42
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/beauty/{training,evaluation,testing}；seed=42。共同目录12101商品，evaluation/testing各22363用户。
* 内容固定输入：wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v5，实际shape=[12101,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3894（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9025（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/beauty-seed42/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29560
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_cf_train; dataset=beauty; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/c184aneu?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/5utubi3y?role=checkpoint&alias=v0&file=step_step=044000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/025s9ets?role=checkpoint&alias=v0&file=step_step=024000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29560
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_sid; dataset=beauty; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29560
bash ./letter_train.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_train; dataset=beauty; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29560
bash ./letter_inference.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-60; LETTER; stage=letter_inference; dataset=beauty; seed=42; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_beauty
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29560
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-60; LETTER CF; dataset=beauty; seed=42; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_beauty
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [22363,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-61

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Beauty
* Seed：200
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/beauty/{training,evaluation,testing}；seed=200。共同目录12101商品，evaluation/testing各22363用户。
* 内容固定输入：wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v5，实际shape=[12101,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3861（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9544（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/beauty-seed200/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29561
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_cf_train; dataset=beauty; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/223be7uk?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/ize3pkzo?role=checkpoint&alias=v2&file=step_step=024000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/6e7ruv7a?role=checkpoint&alias=v2&file=step_step=048000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29561
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_sid; dataset=beauty; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29561
bash ./letter_train.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_train; dataset=beauty; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29561
bash ./letter_inference.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-61; LETTER; stage=letter_inference; dataset=beauty; seed=200; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_beauty
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29561
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-61; LETTER CF; dataset=beauty; seed=200; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_beauty
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [22363,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-62

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Beauty
* Seed：2026
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/beauty/{training,evaluation,testing}；seed=2026。共同目录12101商品，evaluation/testing各22363用户。
* 内容固定输入：wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v5，实际shape=[12101,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3857（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9717（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/beauty-seed2026/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29562
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_cf_train; dataset=beauty; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/5a5gmzoa?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/yhm7t0gv?role=checkpoint&alias=v1&file=step_step=042000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=beauty; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/5m1w7c9r?role=checkpoint&alias=v1&file=step_step=072000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29562
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_sid; dataset=beauty; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29562
bash ./letter_train.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_train; dataset=beauty; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_beauty
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29562
bash ./letter_inference.sh \
  --data-dir data/beauty \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-62; LETTER; stage=letter_inference; dataset=beauty; seed=2026; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_beauty
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29562
bash ./letter_cf_train.sh \
  --data-dir data/beauty \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&alias=v5&file=merged_predictions_tensor.pt' \
  --dataset beauty \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-62; LETTER CF; dataset=beauty; seed=2026; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_beauty
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [22363,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-63

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Sports
* Seed：42
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/sports/{training,evaluation,testing}；seed=42。共同目录18357商品，evaluation/testing各35598用户。
* 内容固定输入：wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v6，实际shape=[18357,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3857（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.8446（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/sports-seed42/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29563
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_cf_train; dataset=sports; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/hsrnapw7?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/pvinh0ml?role=checkpoint&alias=v0&file=step_step=033000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/h5912m4c?role=checkpoint&alias=v1&file=step_step=072000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29563
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_sid; dataset=sports; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29563
bash ./letter_train.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_train; dataset=sports; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29563
bash ./letter_inference.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-63; LETTER; stage=letter_inference; dataset=sports; seed=42; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_sports
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29563
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-63; LETTER CF; dataset=sports; seed=42; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_sports
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [35598,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-64

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Sports
* Seed：200
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/sports/{training,evaluation,testing}；seed=200。共同目录18357商品，evaluation/testing各35598用户。
* 内容固定输入：wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v6，实际shape=[18357,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3872（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9783（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/sports-seed200/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29564
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_cf_train; dataset=sports; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/6los4ob4?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/h3bhzool?role=checkpoint&alias=v1&file=step_step=014000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/6vzov2iw?role=checkpoint&alias=v0&file=step_step=036000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29564
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_sid; dataset=sports; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29564
bash ./letter_train.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_train; dataset=sports; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29564
bash ./letter_inference.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-64; LETTER; stage=letter_inference; dataset=sports; seed=200; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_sports
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29564
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-64; LETTER CF; dataset=sports; seed=200; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_sports
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [35598,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-65

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Sports
* Seed：2026
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/sports/{training,evaluation,testing}；seed=2026。共同目录18357商品，evaluation/testing各35598用户。
* 内容固定输入：wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v6，实际shape=[18357,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3845（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=11.0369（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/sports-seed2026/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29565
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_cf_train; dataset=sports; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### CF导出

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

### Tokenizer训练

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

### SID导出

```bash
export LETTER_CF_EMBEDDING='wandb://baymaxam/GRID/fmxspnds?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt'
export LETTER_CF_BEST_CKPT='wandb://baymaxam/GRID/4kzf622t?role=checkpoint&alias=v2&file=step_step=018000.ckpt'
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=sports; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export LETTER_TOKENIZER_BEST_CKPT='wandb://baymaxam/GRID/lkrugs0y?role=checkpoint&alias=v2&file=step_step=036000.ckpt'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29565
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_sid; dataset=sports; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29565
bash ./letter_train.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_train; dataset=sports; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_sports
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29565
bash ./letter_inference.sh \
  --data-dir data/sports \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-65; LETTER; stage=letter_inference; dataset=sports; seed=2026; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_sports
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29565
bash ./letter_cf_train.sh \
  --data-dir data/sports \
  --embedding-path 'wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&alias=v6&file=merged_predictions_tensor.pt' \
  --dataset sports \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-65; LETTER CF; dataset=sports; seed=2026; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_sports
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [35598,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-66

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Toys
* Seed：42
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/toys/{training,evaluation,testing}；seed=42。共同目录11924商品，evaluation/testing各19412用户。
* 内容固定输入：wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v7，实际shape=[11924,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3868（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9484（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/toys-seed42/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29566
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_cf_train; dataset=toys; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### CF导出

```bash
: "${LETTER_CF_BEST_CKPT:?先填写本单元CF训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29566
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_cf_export; dataset=toys; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Tokenizer训练

```bash
: "${LETTER_CF_EMBEDDING:?先填写本单元CF导出的真实bundle路径或固定版本URI}"
: "${LETTER_CF_BEST_CKPT:?填写CF源checkpoint}"
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=42; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29566
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_tokenizer_train; dataset=toys; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### SID导出

```bash
: "${LETTER_CF_EMBEDDING:?填写已核验CF bundle}"
: "${LETTER_CF_SOURCE:?填写CF来源}"
: "${LETTER_TOKENIZER_BEST_CKPT:?先填写本单元按val/collision_rate选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29566
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_sid; dataset=toys; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29566
bash ./letter_train.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_train; dataset=toys; seed=42; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29566
bash ./letter_inference.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-66; LETTER; stage=letter_inference; dataset=toys; seed=42; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_toys
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29566
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 42 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-66; LETTER CF; dataset=toys; seed=42; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_toys
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [19412,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-67

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Toys
* Seed：200
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/toys/{training,evaluation,testing}；seed=200。共同目录11924商品，evaluation/testing各19412用户。
* 内容固定输入：wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v7，实际shape=[11924,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3881（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=11.0198（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/toys-seed200/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29567
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_cf_train; dataset=toys; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### CF导出

```bash
: "${LETTER_CF_BEST_CKPT:?先填写本单元CF训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29567
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_cf_export; dataset=toys; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Tokenizer训练

```bash
: "${LETTER_CF_EMBEDDING:?先填写本单元CF导出的真实bundle路径或固定版本URI}"
: "${LETTER_CF_BEST_CKPT:?填写CF源checkpoint}"
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=200; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29567
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_tokenizer_train; dataset=toys; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### SID导出

```bash
: "${LETTER_CF_EMBEDDING:?填写已核验CF bundle}"
: "${LETTER_CF_SOURCE:?填写CF来源}"
: "${LETTER_TOKENIZER_BEST_CKPT:?先填写本单元按val/collision_rate选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29567
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_sid; dataset=toys; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29567
bash ./letter_train.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_train; dataset=toys; seed=200; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29567
bash ./letter_inference.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-67; LETTER; stage=letter_inference; dataset=toys; seed=200; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_toys
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29567
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 200 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-67; LETTER CF; dataset=toys; seed=200; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_toys
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [19412,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。

# BMX-68

## 实验单元

* Method：LETTER-TIGER（独立实现）
* Dataset：Toys
* Seed：2026
* 主指标：testing NDCG@10
* 报告指标：Recall/NDCG@5/@10

## 当前状态（2026-10-03）

Todo / 可运行准备完成，完整实验由用户手动启动。BMX-58 提供独立CF→Tokenizer→SID→推荐→Testing链路；本单元尚无正式run、best或指标。未来checkpoint与CF/SID变量由本单元前序阶段实际产物填写，shell守卫拒绝缺失来源，不虚构尚未生成的Artifact。

## 固定协议与输入

* 协议：baseline_protocol=letter-official-grid-v1；CF=letter-cf-sasrec32-grid-v1；评价=full-catalog-no-history-filter-v1。
* 独立 LETTER 模型，不调用项目 RQ-VAE/TIGER/SASRec/LIGER/CoPMRec 模型；公共框架组件可复用。
* CF teacher：32维、history50、2blocks/1head/dropout0.5；逐位置正负 BCE，负样本排除完整 training 行；单卡batch128、FP32、Adam lr0.001/betas(0.9,0.98)、无scheduler，50k step、每1000步 evaluation NDCG10选优。仅 training 参与梯度；不消费 testing。此teacher训练设置为明确的GRID适配，作者未公布完整teacher训练代码。
* Tokenizer：输入1024维共同内容+同目录32维CF，latent32、4×256、10groups；重建/VQ/CF/diversity，alpha0.01、beta0.0001、mu0.25；AdamW lr0.001/WD0.0001、batch1024、单卡、最多20000 epoch，每2000 epoch完整目录 collision_rate 最小选优；constrained K-means n_init10/max_iter10/n_jobs10，固定seed；SID末层Sinkhorn最多20轮，容量内残余碰撞用全prefix末层最小距离一对一分配；prefix人口超过256则拒绝输出，不增加第五位。该硬匹配为GRID导出适配，协议sid_export_protocol=letter-sinkhorn-prefix-assignment-v1。
* 推荐：history20、所有非空历史前缀监督下一商品；T5随机初始化128/1024/4layers/6heads/d_kv64/dropout0.1；temperature=1.0（此配置不主张温度增强收益），beam20/top10，全目录Trie且无历史过滤。
* 推荐训练：两卡每卡128/global256、FP32、accumulation1、50k optimizer step、每500步完整 evaluation；AdamW lr0.0005/WD0.01、bias/LayerNorm不decay、warmup500/cosine到0、clip1；仅 val/ndcg@10 最大选best，run_test_after_training=false。
* 固定单组参数，无新增调参搜索预算。统一入口沿用项目matmul_precision=medium；初始化n_jobs10为当前冻结设置，与早期串行结果不作数值等价声明。冷目标及全部有效用户保留；Testing报告Recall/NDCG@5/@10/user_count。keys=用户key、predictions=原始商品key[users,10]。
* 数据：data/toys/{training,evaluation,testing}；seed=2026。共同目录11924商品，evaluation/testing各19412用户。
* 内容固定输入：wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt；Artifact=sem_embeds_inference-semantic-embedding:v7，实际shape=[11924,1024]。CF读取该bundle的keys，不读取内容向量训练teacher。
* CF、tokenizer、SID、推荐checkpoint按本单元seed独立生产。上游CF导出未归一化商品表，包含冷商品；冷商品无正交互监督，可能收到负采样梯度，不宣称具备充分协同信号。不能投影50维基线SASRec或消费RKMeans SID作为LETTER正式输入。
* 使用本单元实际生产者和固定Artifact版本填写变量；检查selection=best、monitor、mode及checkpoint内分数。CF/推荐按val/ndcg@10 max，tokenizer按val/collision_rate min；last只用于恢复。

## 已完成的单元准备验证

node1既有PyTorch2.9.1+cu128/CUDA12.8/Lightning2.6.5；本单元CF训练dry-run：物理GPU7、batch128/4workers/FP32，exit0、max_steps1，loss=1.3888（有限，非性能）。推荐训练dry-run：物理GPU6/7、NPROC2、每卡batch128/global256、4workers、FP32，exit0、max_steps1，loss=10.9862（有限，非性能）。统一入口禁用dry-run validation/testing及业务发布。

三个真实目录的32维CF有限导出、4×256初始化探针和native SID均通过；本单元推荐dry-run使用对应数据集seed42零学习率初始化探针SID，仅验证入口，不代表本单元seed的正式CF/Tokenizer已训练。SID合法唯一、前三码保留，同CUDA/medium环境重复输出一致；CPU/CUDA不承诺数值相同。早期五步Tokenizer未收敛、prefix超256而拒绝导出的失败保留；正式训练后仍必须重新检查SID，不能使用验证产物替代。

Beauty真实目录额外完成两卡5步训练、step5恢复和evaluation预测128×10，checkpoint/optimizer step5及输出合法唯一、独立目标/指标复核通过；未消费testing、不登记正式性能。62项聚焦测试、九槽位63命令块stub、Ruff与七个OpenSpec strict通过；完整证据见BMX-58和docs/letter-readiness-20261003.md，日志为node1:logs/letter-readiness-20261003/toys-seed2026/{cf-dry,recommendation-dry}/。Mutagen三session Watching for changes无conflict。

## 从仓库根目录启动

node1:/data3/weizhenyu/projects/GRID。各阶段独立执行；更换物理GPU可修改该命令的CUDA_VISIBLE_DEVICES，NPROC_PER_NODE及逻辑devices由脚本保持一致。各阶段逐参数换行，显式MASTER_PORT，notes含issue/dataset/seed，额外Hydra override放在末尾覆盖默认。每次交付新代码先执行Mutagen flush。CF和推荐分别50k step，Tokenizer最多20000 epoch；以下正式命令不带dry-run。

### CF teacher训练

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29568
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_cf_train; dataset=toys; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### CF导出

```bash
: "${LETTER_CF_BEST_CKPT:?先填写本单元CF训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29568
bash ./letter_cf_export.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --ckpt-path "$LETTER_CF_BEST_CKPT" \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_cf_export; dataset=toys; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Tokenizer训练

```bash
: "${LETTER_CF_EMBEDDING:?先填写本单元CF导出的真实bundle路径或固定版本URI}"
: "${LETTER_CF_BEST_CKPT:?填写CF源checkpoint}"
export LETTER_CF_SOURCE="protocol=letter-cf-sasrec32-grid-v1; dataset=toys; seed=2026; gradient_split=training; selection_split=evaluation; dim=32; checkpoint=$LETTER_CF_BEST_CKPT"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29568
bash ./letter_tokenizer_train.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_tokenizer_train; dataset=toys; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### SID导出

```bash
: "${LETTER_CF_EMBEDDING:?填写已核验CF bundle}"
: "${LETTER_CF_SOURCE:?填写CF来源}"
: "${LETTER_TOKENIZER_BEST_CKPT:?先填写本单元按val/collision_rate选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29568
bash ./letter_sid.sh \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --cf-embedding-path "$LETTER_CF_EMBEDDING" \
  --cf-source "$LETTER_CF_SOURCE" \
  --ckpt-path "$LETTER_TOKENIZER_BEST_CKPT" \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_sid; dataset=toys; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys \
  input_dim=1024
```

### 推荐训练

```bash
: "${LETTER_SID:?先填写本单元LETTER导出的唯一四码SID bundle路径或固定版本URI}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29568
bash ./letter_train.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_train; dataset=toys; seed=2026; manual formal pipeline' \
  logger.wandb.group=paper_main_letter_toys
```

### Testing

```bash
: "${LETTER_SID:?填写与训练完全相同的SID}"
: "${LETTER_BEST_CKPT:?先填写本单元推荐训练按val/ndcg@10选出的真实best引用}"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29568
bash ./letter_inference.sh \
  --data-dir data/toys \
  --semantic-id-path "$LETTER_SID" \
  --ckpt-path "$LETTER_BEST_CKPT" \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --notes 'issue=BMX-68; LETTER; stage=letter_inference; dataset=toys; seed=2026; testing; metric-selected checkpoint' \
  logger.wandb.group=paper_main_letter_toys
```

### 显式 dry-run

CF训练、Tokenizer训练、推荐训练均支持在上述对应命令的末尾追加 `--dry-run`；不会默认启用。以下 CF dry-run 可直接执行；后两阶段先产生本单元CF/SID，再追加 `--dry-run` 验证，不能用其他量化器SID替代。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export MASTER_PORT=29568
bash ./letter_cf_train.sh \
  --data-dir data/toys \
  --embedding-path 'wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt' \
  --dataset toys \
  --seed 2026 \
  --gpus "$CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT" \
  --dry-run \
  --notes 'issue=BMX-68; LETTER CF; dataset=toys; seed=2026; non-benchmark dry-run' \
  logger.wandb.group=paper_main_letter_toys
```

## 当前正式结果

尚未运行。准备检查及有限训练产物只验证链路，不登记正式性能，不作为本单元正式上游。正式训练后填写CF导出、tokenizer/SID/推荐run与固定版本，Testing只消费本单元val-selected推荐best。

## Done 门禁

准备依赖已解除且命令通过验证；各正式上游与推荐训练/Testing run均finished，resolved config符合冻结协议；CF训练split/seed/32维目录、SID唯一性及生产链路、metric-selected checkpoint、原始数据用户/目标/目录与运行时源码可追溯；Testing bundle [19412,10]逐用户合法且无重复，first-match指标与W&B summary独立复算一致。负向但协议有效的结果仍视为有效。
