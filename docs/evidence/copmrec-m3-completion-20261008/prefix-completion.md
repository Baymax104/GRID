## 1. 协议与状态

`copmrec-m3-v53-20261008-v1`；experiment_id=M3；类型=checkpoint_diagnosis；base_release=copmrec-v5.3。Done；implementation=runtime_validated；launch_authorization=true（用户 2026-10-08 授权）；started/completed_sources=3/3；planned_sources=3；results=Full/A1/A4 observed。目的为论文方法实证，不判别好坏、不预设积极/消极结果。Done 只按本 issue 的证据交付完整性，与差值方向/显著性无关；父 BMX-117 和整个 milestone 的状态另行核对。[完整协议](https://linear.app/baymax104/document/copmrec-v53m3-%E6%B6%88%E8%9E%8D%E4%B8%8E%E6%9C%BA%E5%88%B6%E5%AE%9E%E8%AF%81%E5%8D%8F%E8%AE%AE2026-10-08-643847ec8183)。

## 2. 实证问题与解释边界

测量真目标前缀下生成概率、内容mass与混合概率的条件行为及移除/替换前缀监督后的数值差。该观测不是free-running候选可达率，不推断dense的beam恢复或推理效率。 固定代表数据集Beauty及training seed42，不根据结果换单元；整体多seed验证由主矩阵承担，本消融只描述该单元的观测。

## 3. 对照与干预

Beauty/seed42 Full/A1/A4各1个own-best，全部Testing用户、4层真实目标前缀teacher forcing；同合法兄弟mask比较p_gen/p_mass，Full p_mix=(1-alpha)p_gen+alpha\*p_mass及checkpoint全局alpha。A1/A4 gate未学习，mixed观测字段=null，不伪装成有效混合模型。使用未历史屏蔽的joint logits构造训练同定义mass支持；正式预测仍历史排除且不得读取标签。

## 4. 数据、seed与来源

Beauty / training seed42，只有1个固定单元；SID `wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt`，Artifact `rkmeans_inference_beauty-semantic-id:v0`，digest `19ace08283163fbe687287fbacaa3842`；content `wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt`，Artifact `sem_embeds_inference-semantic-embedding:v5`，digest `ab56af975eac589c27eed6094482cbd7`。目录为 node1 `data/beauty`；训练/Testing 目录、TFRecord 文件 SHA、训练频次、历史长度和目标 SID 均核对。共同用户/标签 SHA256=`55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`；12,101 catalog items；各来源 22,363 用户。

| 来源 | 训练 run / own-best step | 不可变 checkpoint URI | 文件 SHA256 |
| -- | -- | -- | -- |
| Full / [BMX-120](https://linear.app/baymax104/issue/BMX-120) | gshpyn49 / 47500 | `wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt` | `a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4` |
| A1 / no_mixture / [BMX-122](https://linear.app/baymax104/issue/BMX-122) | m3frgiim / 48500 | `wandb://baymaxam/GRID/m3frgiim?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=048500.ckpt` | `35cb99260fc4a08ff54844954b714b286a18ae85fe512bc6e74697a87a60da71` |
| A4 / legal_generation / [BMX-142](https://linear.app/baymax104/issue/BMX-142) | m3odfrrh / 46000 | `wandb://baymaxam/GRID/m3odfrrh?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=046000.ckpt` | `15e36379dda9cf6df7f7192b9c9e2b26ac672b990bec10b4d1672febeef2ad0e` |

A1/A4 均由各自 raw val/ndcg@10 的首次最高值选点；checkpoint Artifact selection=best、global_step、文件 SHA、strict CPU 恢复及输入身份终验已通过（`docs/evidence/copmrec-m3-completion-20261008/training-audit-summary.json`）；未用 last.ckpt 或 Full 初始化。A1/A4 历史训练源码 SHA=`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`。三个 diagnosis 的运行字节源码 SHA=`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`（304文件、origin verified）；两版本仅 diagnosis/writer 的嵌套 metadata 序列化不同，模型计算及恢复调用链字节一致，分别登记，不覆盖历史来源。所有 own-best 与 SID/content 的 consumed Artifact lineage 已核对。

## 5. 固定协议

Beauty/seed42，匹配正式输入/公共参数初值及数据顺序；DDP2/global256/FP32/50k、主lr0.0003/残差lr0.002/wd0.035、warm2500/cosine50000；每500步raw dense val/ndcg@10各臂选自己的首次best。单卡Testing使用joint full-catalog cosine/0.07、最近20件输入历史排除、cold eligible、stable catalog row Top10。额外独立Validation=0。诊断沿用来源checkpoint，仅eval、无更新、无重新选点。

## 6. 执行步骤与命令

三来源均已在 node1 从仓库根目录通过统一入口的单卡 Trainer.test 完成，exit_code=0、W&B state=finished。物理 GPU1 → CUDA local[0]，NPROC=1，group=`paper_mechanism_copmrec_beauty`。Full `j2rworuj` 复用此前完成诊断；此次只顺序执行 A1 和 A4 两个独立 tmux，没有重复 Full forward。

| 来源 | diagnosis run | tmux |
| -- | -- | -- |
| Full | j2rworuj | `copmrec_prefix_full_bmx146_20261008` |
| no_mixture | 5murlou7 | `copmrec_prefix_no_mixture_bmx146_5murlou7` |
| legal_generation | pltnvpjf | `copmrec_prefix_legal_generation_bmx146_pltnvpjf` |

以下为各来源实际执行命令，own-best URI/SHA、seed、GPU、run ID 和输出目录明确固定。

### 来源 full（BMX-120）

```bash
export CUDA_VISIBLE_DEVICES=1
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
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=full; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations' \
  +logger.wandb.id='j2rworuj' \
  hydra.run.dir=logs/copmrec_m3_launch_20261008/prefix-full_j2rworuj/hydra
```

### 来源 no_mixture（BMX-122）

```bash
export CUDA_VISIBLE_DEVICES=1
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='wandb://baymaxam/GRID/m3frgiim?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=048500.ckpt'
export COPMREC_CHECKPOINT_SHA256='35cb99260fc4a08ff54844954b714b286a18ae85fe512bc6e74697a87a60da71'

bash ./copmrec_diagnosis.sh \
  --analysis prefix --variant no_mixture \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=no_mixture; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations' \
  +logger.wandb.id='5murlou7' \
  hydra.run.dir=logs/copmrec_m3_completion_20261008/prefix_no_mixture_5murlou7/hydra
```

### 来源 legal_generation（BMX-142）

```bash
export CUDA_VISIBLE_DEVICES=1
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='wandb://baymaxam/GRID/m3odfrrh?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=046000.ckpt'
export COPMREC_CHECKPOINT_SHA256='15e36379dda9cf6df7f7192b9c9e2b26ac672b990bec10b4d1672febeef2ad0e'

bash ./copmrec_diagnosis.sh \
  --analysis prefix --variant legal_generation \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-146; protocol=copmrec-m3-v53-20261008-v1; analysis=prefix; source_variant=legal_generation; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations' \
  +logger.wandb.id='pltnvpjf' \
  hydra.run.dir=logs/copmrec_m3_completion_20261008/prefix_legal_generation_pltnvpjf/hydra
```

按真实目标前缀测量逐层 gen/mass 的概率、NLL、目标 rank、熵、JS 与 argmax 一致率；Full 另测 mixed 及全局 alpha，A1/A4 的 mixed/alpha 字段 null。预定义概率差分箱、训练频次/seen/历史切片，保留空组和实际 N。共享 writer 登记最终观测，诊断仅 eval，无优化器更新、重新选点或部署改动。

## 7. 观测指标与统计

每层target probability/NLL、合法target rank、entropy、gen/mass JS divergence、argmax一致率；各depth与seen/cold×depth的N/mean/quantiles。Full全局alpha与已有loss日志；p_mass(target)-p_gen(target)连续散点及固定\[-1,-0.1,0,0.1,1\]区间，不按Testing重选。缺失日志=null；无用户级gate解释。

## 8. 成本与依赖

本 issue 实际 0train/0updates/0额外Val/0正式Testing；3个 checkpoint → 3个 Trainer.test diagnosis 任务/3个全量 forward pass，已完成 Full1+A1 1+A4 1。本次继续执行新增2个 diagnosis；Full已完成分支复用。资源观测：Full W&B runtime=56秒、Testing进度=49秒；A1 W&B runtime=50秒、Testing进度=41秒；A4 W&B runtime=49秒、Testing进度=39秒。GPU小时=null（未记录完整分配时间），CUDA allocator 峰值显存=null（未采样）；不由 W&B runtime 换算或填补。

来源依赖：[BMX-116](https://linear.app/baymax104/issue/BMX-116)、[BMX-122](https://linear.app/baymax104/issue/BMX-122)、[BMX-142](https://linear.app/baymax104/issue/BMX-142)；A1/A4 的 own-best 训练/来源审计已通过，执行顺序不取决于观测方向。M3固定累计5train/250,000updates/0额外Val/5Testing，另4diagnosis任务/6全量forward等价pass；Full1组复用。该预算登记不替代父任务完成审计，不增加多seed、多数据集或按结果扩预算。

## 9. 结果登记

| Source | Run | Completion | Own-best / 最终分析 Artifact / digest |
| -- | -- | -- | -- |
| Full | [j2rworuj](https://wandb.ai/baymaxam/GRID/runs/j2rworuj) | finished；独立核验通过 | gshpyn49 / 47500；`copmrec_beauty_seed42_prefix_full_diagnosis-analysis:v0`；`8affd7f6aa60964339f257a6cbc44465` |
| A1 / no_mixture | [5murlou7](https://wandb.ai/baymaxam/GRID/runs/5murlou7) | finished；独立核验通过 | m3frgiim / 48500；`copmrec_beauty_seed42_prefix_no_mixture_diagnosis-analysis:v0`；`eace7e07f3dd2692e6e63c3377858293` |
| A4 / legal_generation | [pltnvpjf](https://wandb.ai/baymaxam/GRID/runs/pltnvpjf) | finished；独立核验通过 | m3odfrrh / 46000；`copmrec_beauty_seed42_prefix_legal_generation_diagnosis-analysis:v0`；`60b426d9d42205599bf01c40f3f5ce58` |

每来源 22,363 个唯一用户 × 4 depth = 89,452 条条件化观测；三来源共同用户及标签完全对齐。下表为全部用户的原始均值，N 为该层用户数。

| Source | Depth | N | p_gen(target) | p_mass(target) | p_mix(target) | gen NLL | mass NLL | mixed NLL | JS | argmax agree |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Full | 1 | 22363 | 0.04147758 | 0.04453813 | 0.04449451 | 4.83247517 | 4.90343844 | 4.89753096 | 0.04988156 | 0.51992130 |
| Full | 2 | 22363 | 0.13648378 | 0.15595493 | 0.15567743 | 3.24297780 | 3.52645088 | 3.45775973 | 0.18707240 | 0.37262442 |
| Full | 3 | 22363 | 0.71889938 | 0.72388059 | 0.72380960 | 0.95684000 | 0.78053831 | 0.76140299 | 0.07795790 | 0.77051380 |
| Full | 4 | 22363 | 0.90931486 | 0.91408727 | 0.91401925 | 0.22342529 | 0.21595496 | 0.20882411 | 0.02011673 | 0.92585968 |
| A1 | 1 | 22363 | 0.04214875 | 0.04465227 | null | 4.86367109 | 4.92317942 | null | 0.04711744 | 0.53937307 |
| A1 | 2 | 22363 | 0.13556747 | 0.15532262 | null | 3.25118191 | 3.50520466 | null | 0.18587906 | 0.38000268 |
| A1 | 3 | 22363 | 0.71715984 | 0.72338974 | null | 0.94896961 | 0.77633388 | null | 0.07717160 | 0.77297321 |
| A1 | 4 | 22363 | 0.90967481 | 0.91417765 | null | 0.22609570 | 0.21288534 | null | 0.01965290 | 0.92988418 |
| A4 | 1 | 22363 | 0.04198409 | 0.04449446 | null | 4.89980302 | 4.95167513 | null | 0.04200882 | 0.56445915 |
| A4 | 2 | 22363 | 0.13156769 | 0.15306473 | null | 3.14498638 | 3.48270616 | null | 0.16898461 | 0.39538523 |
| A4 | 3 | 22363 | 0.71217001 | 0.72293039 | null | 0.75257331 | 0.76981923 | null | 0.06293777 | 0.77981487 |
| A4 | 4 | 22363 | 0.90915900 | 0.91379942 | null | 0.20702913 | 0.20962817 | null | 0.01790882 | 0.92773778 |

Full global alpha=`0.9857481122016907`，为 checkpoint 全局参数。A1/A4 checkpoint 的 gate_logit=0 且被冻结；其 alpha 及 mixed probability/NLL/rank/entropy 观测全部 null，不登记成学习后的混合概率。Full p_mix 公式最大绝对误差=`1.5516809881432891e-7`。观测均来自真实目标前缀 teacher forcing、训练同定义的未排除历史目录 mass 支持；不据此推断 free-running 候选可达率、beam恢复或方法优劣。

全部 gen/mass target 概率与 NLL 关系、合法 rank/entropy/JS 范围、固定差分箱、4层完整 keys、seen/cold/训练频次/历史/seen×history 的 N/均值/quantiles/空组均经用户级输出独立复算；A1/A4 各92条 slice 统计及 null 字段核验通过。训练频次分位阈值=[5,10]，difference bins=[-1,-0.1,0,0.1,1]，未按结果重新分组。Full 原独立核验复用。共同 user-label SHA 见第4章；每个运行的 source archive、源文件逐项 SHA、数据 manifests、consumed checkpoint/SID/content URI/digest、own-best SHA、release variant 与输出状态均核对。

三个 analysis Artifact 各有8个 file:// reference；云端 manifest 的文件名、ref实际绝对路径、size 与 base64-MD5 均与 node1 最终完成文件匹配，并登记各文件 SHA256。这些是最终 node1 文件引用，文件字节未上传成 W&B 文件副本。prefix 不使用的 cases/catalog_norms/catalog_norm_slices/comparisons 保留0字节；实际用户/切片/summary/manifest完整，未以文件数量代替身份核验。

| 来源 | node1 最终目录 | 本地独立审计回执 | runtime source archive SHA256 |
| -- | -- | -- | -- |
| Full | `/data3/weizhenyu/projects/GRID/logs/copmrec_m3_launch_20261008/prefix-full_j2rworuj/hydra/analysis` | `docs/evidence/copmrec-m3-launch-20261008/prefix-full-independent-audit.json` | `38f10df2c9ec29e2ea483cd0ad4b3995ebb83a3b4c66b06cec5b0015c4818003` |
| A1 | `/data3/weizhenyu/projects/GRID/logs/copmrec_m3_completion_20261008/prefix_no_mixture_5murlou7/hydra/analysis` | `docs/evidence/copmrec-m3-completion-20261008/prefix-no_mixture-independent-audit.json` | `2fe1965e79fa7468f71d483cec1b004e3aae44eac683904d5ef0a9c5a937d654` |
| A4 | `/data3/weizhenyu/projects/GRID/logs/copmrec_m3_completion_20261008/prefix_legal_generation_pltnvpjf/hydra/analysis` | `docs/evidence/copmrec-m3-completion-20261008/prefix-legal_generation-independent-audit.json` | `9dcbc5d472340a26e17b1199e4d30b39fcca5e7107c6cbaeb7c6cb0899bd329a` |

三来源表/图位于 `docs/evidence/copmrec-m3-completion-20261008/`：`prefix-three-source-values.csv`（12行、各观测 mean/q10/q50/q90、rank/entropy/N）、`prefix-three-source-summary.json`（各来源身份与 A1−Full/A4−Full/A4−A1 有符号均值差）、`prefix-three-source-observations.png/.pdf`（三来源逐层原值）、`prefix-target-probability-differences.png/.pdf`（3来源×4层全量用户的 p_gen(target) 与 p_mass(target)−p_gen(target) 散点）。CSV复制经过各源 Artifact entries 的 size/SHA 比对（`prefix-table-collection.json`）；两个图已视觉核验可读。全 seen/cold、频次、历史、固定差分箱、空组及 quantiles 保留在各 run 的 users/slices/summary 和独立审计，不挑选用户或有利区间。现有训练 loss 日志保留来源训练 run 链接；缺失独立资源/日志字段保持 null。

## 10. 交付标准

预定固定单元、全部对照与统计字段完整；输入/源代码/own-best/输出可追溯，计算独立核对；全量表及图回溯到用户级数据，缺失项和执行异常显式保留。证据完成后Done，与观测方向、显著性或是否支持预期叙事无关。
