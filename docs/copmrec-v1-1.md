# CoPMRec v1.1：批量候选评分与运行命令

日期：2026-10-03。v1.1已实现。它保留v1的模型目标和评价协议，改进候选评分的执行方式；推荐收益仍需完整训练和Recall/NDCG评价。

2026-10-04更新：用户采纳training-only校准的排序权重，v1.1组件默认值改为 `0.05726763550972437`。下方训练命令已更新为从零初始化，显式关闭checkpoint恢复与v0预训练；上一轮短程续训仅用于验证，结果见 [权重验证](copmrec-loss-weight-verification.md)。v1组件与模型Python API默认值仍为1；旧等权v1.1 checkpoint恢复/推理须显式传 `model.root.ranking_loss_weight=1`。

## 实现与版本

- 模型：`src.recommendation.liger.batched_relevance.BatchedRelevanceCoPMRec`，继承v1共享主干、候选规则、四项loss与trace。
- 训练和hybrid推理将各用户候选拼接为`(user,catalog_row)`，每块最多256对，批量调用decoder，然后按原用户列表拆回。通过encoder状态索引保留autograd，全部推荐参数可训练。
- 完整候选SID、beam20、全部cold、训练content Top20与正例、每rank排序抽样4例、score、selection/audit、validation频率和更新预算沿用v1。
- v0、v1入口继续保留。v1.1根脚本与Hydra名字为`copmrec_v1_1_train`、`copmrec_v1_1_inference`；W&B配置字段`copmrec_version=v1.1`，group为`copmrec_v1_1`。
- 本次批量化不涉及T5内部cross-attention K/V复用；该项需要更深入的实现和单独验证。

`candidate_chunk_size`在v1中限制单用户候选块，在v1.1中限制整个batch的候选对块。v1.1 checkpoint记录版本、批量执行方式和chunk大小，恢复时须一致；不直接以v1 checkpoint恢复v1.1。既有v0 weights-only预训练仍可选，载入后全部参数继续训练。下面命令默认从零训练。

批量化改变dropout随机数分配和浮点归约顺序。关闭dropout时进行评分/loss/梯度对照；不承诺与v1同seed训练轨迹逐位相同。

## 性能与验证

node1 A100 80GB物理GPU1，真实Beauty training preprocessing，batch128、80 SID token、12101商品、排序用户4。使用同一组随机初始化权重，并将head输出权重设为小的非零值以覆盖残差上游梯度；各预热3次，轮换顺序交替计时12次。仅比较候选评分执行和chunk，不执行optimizer更新。

| 路径 | 前向＋反向中位数 | 峰值allocated |
|---|---:|---:|
| v1：逐用户，chunk64 | 345.85 ms | 3.113 GiB |
| v1.1：批量，chunk64 | 267.93 ms | 2.990 GiB |
| v1.1：批量，默认chunk256 | 191.93 ms | 2.996 GiB |

默认v1.1耗时减少约44.5%，同一探针内约1.80倍加速。保持chunk64也有约22.5%的耗时减少，说明效果同时来自跨用户合并和更大的全局块。完整更新的decoder调用由13次变为7次，候选评分部分由8次变为2次。

关闭dropout的生产规模核验：loss最大绝对差0，全部157组参数梯度有限，参数梯度最大绝对差`1.55e-6`。8用户hybrid评分中位数由129.54 ms变为79.87 ms，Top10 ID完全相同、候选合法且无重复。这8个用户取自训练历史，不是完整selection/audit评价。

原始标量及逐次计时见 [production-probe.json](evidence/copmrec-v1-1-20261003/production-probe.json)，实际inline诊断源码见同目录 `production-probe-source.txt`。本探针没有optimizer state、DDP通信或完整用户验证；GPU1当时另一进程有驻留显存、计算采样为0%。实测不能直接换算完整多卡训练时间，也不构成推荐收益证据。

内存单元测试覆盖不等长/空列表、尾块、完整SID、用户映射、确定性全参数loss/梯度、非零dropout两次更新和主干排序梯度、标签独立、trace、checkpoint/optimizer恢复。配置脚本检查覆盖Hydra组合、LF、语法、notes两种形式、quoting、错误/空参数、override、torchrun和默认非dry-run。node1两进程CPU Gloo回归连续2次更新后全部参数一致，包含非等长用户分片的全局指标分母。

本地模型/trace/检索/配置回归65项通过、2项Linux Gloo跳过；最后配置脚本检查31项通过（与前者重叠30项，新增chunk override回归1项）。node1聚焦回归55项通过，包含v1与v1.1两进程训练；最终两条v1.1 Linux Bash入口的双进程参数检查通过。Ruff check/format和OpenSpec strict通过。核验记录见 [verification.json](evidence/copmrec-v1-1-20261003/verification.json)。

## 完整训练命令

从node1仓库根目录执行。下例沿用此前Beauty输入和GPU2、3；这些卡在排查时有活动进程，执行前应按实际可用资源调整可见GPU。逻辑`--devices '[0,1]'`无需随物理编号改变。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v1_1_train.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29544 \
  --group copmrec_v1_1_calibrated_scratch \
  --notes 'CoPMRec v1.1：校准排序权重0.0572676，从零初始化全部推荐参数，固定selection/audit，50k更新' \
  model.root.candidate_chunk_size=256 \
  model.root.ranking_loss_weight=0.05726763550972437 \
  model.root.allow_ranking_loss_reweighting=false \
  ckpt_path=null \
  pretrained_checkpoint_path=null
```

默认50k optimizer更新、每卡batch128、累积1、有效batch256、每rank每微批4个排序用户、每2500微批validation、FP32。脚本通过`uv run torchrun --nproc_per_node=2 ... -m src.main experiment=copmrec_v1_1_train`运行，默认`UV_NO_SYNC=1`，不默认dry-run。需要有界smoke时显式追加`--dry-run`；不能把smoke结果当作正式推荐收益。

本次新的从零完整训练由用户手动开始，未由agent启动；此前用户启动的等权run `f91njtjx` 不受此命令交付影响。运行源快照、resolved配置、上游lineage及best checkpoint由统一launcher与既有callbacks记录。

## 推理与恢复

完成训练后，选取v1.1 run实际发布的best checkpoint Artifact（`selection=best`、`monitor=val/ndcg@10`、`mode=max`），使用`copmrec_v1_1_inference.sh`并传同样的数据、SID、embedding、devices与`--checkpoint "$COPMREC_V1_1_BEST_CKPT"`。本次尚未产生该best引用，不能以v1或last checkpoint代替。

恢复本版本训练时用真实v1.1引用设置`ckpt_path`，chunk须与原训练一致。若显存需要更小全局块，可在**新训练**命令设置`model.root.candidate_chunk_size=128`或64；原训练恢复时保持原契约。

## 交付核验

源代码和配置通过既有三会话Mutagen边界交给node1；最终flush/status与运行文件SHA256核验记录在同目录 `runtime-files.json`。完整GPU DDP吞吐与optimizer显存、最终Recall/NDCG仍未验证。当前v1生产进程未被停止或重启。
