# MIR 完整方法实验方案与启动命令

日期：2026-09-13。协议：`mir-v1`。当前状态：独立模块已实现，CPU/配置/脚本检查通过；完整 GPU 实验尚未启动。

> 2026-09-14 更新：用户已停止并删除首次运行，validation 性能修复现已完成。评估跳过无用证据，正式 trace 上界批量化；105 项测试通过，CPU 目录规模搜索约快 16 倍，真实 GPU 速度待确认。请同步修复后按本文原命令从头重启；参数、选择协议与 54-run 矩阵不变。验收及重启记录见 [性能修复验收](../fix-item-resolution-validation-performance/verification.md)。上句“尚未启动”为初版记录。

## 1. 本轮回答什么

核心问题：查询相关的多深度 item 解析，在相同目录、内容输入、训练更新预算下，是否优于固定解析层和强内容基线？总体 NDCG@10 是主指标，实际单卡成本是约束；Head/Mid/Tail/Cold 全部报告，长尾改善不是预设成功条件。

本轮不再追加旧 CGBS continuation。所有条件从头训练，使用 Beauty/Sports × seeds 42、2024、2025 × 9 条件，共 **54 次训练**。两条人工队列各 27 次，各占两张 GPU；每条队列包含两数据集和全部条件的一部分，避免把大数据集全部压在一条队列。

## 2. 条件与需要排除的解释

| arm | 结构/目标 | 用途 |
|---|---|---|
| `mask_ce` | 合法前缀 CE，随机 token 初始化 | 与既有 mask CE 对齐的基础控制 |
| `token_content_init` | 内容初始化，合法前缀 CE | 排除收益仅来自 token 初始化 |
| `dense` | 相同历史 encoder/decoder BOS 状态、内容 query、全目录 item CE | 检验小目录是否根本不需要树路由 |
| `hybrid` | 合法 SID CE + 0.1 全目录内容 CE，生成/dense 候选并集与概率混合 | 本地匹配 hybrid 控制 |
| `cobra` | raw SID CE + 前缀条件全目录 dense CE；raw SID beam 后局部解析和 BeamFusion | COBRA 机制适配控制，非官方复现 |
| `earliest` | 第一个容量允许的前缀直接解析 | 排除“早点做 item 分类就足够” |
| `depth2` | 第二个语义层解析；单 item 桶共同提前结束 | 排除“固定二级 softmax 就足够” |
| `depth_gate` | 多出口边缘似然，gate 仅依赖深度 | 排除收益仅来自固定多深度混合 |
| `mir` | gate 依赖历史/前缀状态，统一 item 边缘似然 | 论文候选主方法 |

除 mask CE 外，所有 arm 使用相同 token 内容初始化；解析/dense/hybrid/COBRA 使用相同冻结 PCA 内容输入和可训练共享投影，无每 item 自由参数。此处 hybrid 是当前匹配控制，不能与旧 20k CGBS hybrid checkpoint 混用。不同目标的绝对 loss 数值不可直接横比。

MIR 与三个解析控制使用同一局部解析器、容量 128、温度 0.1。默认前 2000 步预热路由和可行出口，然后使用 item NLL + 0.1 路由 CE + 0.1 出口 CE。固定容量不允许通过采样真值或删除目录后代满足。

## 3. 固定训练与选择协议

- 训练：Adam，lr=0.0005，weight_decay=0.000001，40,000 optimizer steps，无 scheduler；每卡 batch 128，两卡全局 batch 256。
- 每 500 步在 `evaluation` 验证，完整验证集，保存 best `val/ndcg@10` 和 last。验证 batch 默认每卡 16。
- 原 20k runs 仅作历史证据，不纳入本轮同预算比较；40k 也不保证全部模型收敛，须同时检查 last 与曲线。
- 主结果使用 evaluation 选出的 best checkpoint；last 用于稳定性说明，不能事后在 best/last 之间为每个 arm 挑赢家。
- `testing` 在方法、预算、比较规则冻结之后统一执行；evaluation 不称为独立测试集。
- 不按某个 seed 结果追加短 continuation 或重新选择主指标。修改协议必须另记版本，不能与 `mir-v1` 混合汇总。

初始主决策门槛：MIR 的三 seed 平均 NDCG@10 在两数据集均比最强简单解析控制至少高 2%（相对值），同时报告每 seed 差值、mean/std 和逐用户配对区间。它还必须对 depth_gate 有可见收益，并对强 dense/hybrid/COBRA 提供独立价值；只有比原 TIGER 高不算通过。

成本约束：相同 GPU、batch、数据顺序下，完整模型计算时间不得超过相应最强简单解析控制的 1.5 倍。2% 和 1.5 倍是预声明研究门槛，不是已实现表现或统计显著性的替代物。若准确率/成本只有单边获益，按预设质量—成本报告，不事后改成长尾或加速论文。

## 4. 来源与运行约定

从 GRID 仓库根目录运行，统一入口 `uv run ... -m src.main` 或 `uv run --module src.main`。不调用独立 Python runner。

| 数据集 | 数据根目录 | Semantic ID run | 内容 embedding run |
|---|---|---|---|
| Beauty | `data/beauty` | `wandb://4vyi4o6w` | `wandb://3jtt9mpa` |
| Sports | `data/sports` | `wandb://3narllqy` | `wandb://psec3u5i` |

来源延续已审计 CGBS 矩阵，队列默认固定以上 run URI；可以用 `--beauty-semantic-id-path`、`--sports-semantic-id-path`、对应 embedding 选项显式替换，但替换后属于新设置。实际解析的 Artifact version、来源 run 与 resolved Hydra config 由共享 lineage/logger 保存。

目录下必须有 `training/`、`evaluation/`、`testing/`，训练暂不读取 testing。完整实验由用户启动。**所有 inference 和 calibration 都只使用一张 GPU。**

## 5. 立即可运行的两组训练

终端一，GPU 0–1：

```bash
bash ./tiger_item_resolution_suite.sh \
  --queue 1 \
  --beauty-data-dir data/beauty \
  --sports-data-dir data/sports \
  --notes "MIR v1 full method comparison; queue 1"
```

终端二，GPU 2–3：

```bash
bash ./tiger_item_resolution_suite.sh \
  --queue 2 \
  --beauty-data-dir data/beauty \
  --sports-data-dir data/sports \
  --notes "MIR v1 full method comparison; queue 2"
```

默认包含全部三个 seeds，各队列 27 次训练；失败立即停止该队列。若实际数据根目录不同，只替换两个 `--*-data-dir`。

在相同命令末尾加 `--print-only` 只打印命令，不实例化模型、不访问 W&B、不启动 GPU。`--dry-run` 则是统一入口的小规模执行模式，会实际读数据，二者用途不同。

中断后通过 `--seeds 2024,2025` 或 `--arms mir,depth2` 选择**尚未运行的条件**，它会重新分配所选条件到两组队列；预览后再启动。它不是自动恢复器，也不会替用户判断 W&B 中哪些 run 已完成。单条件可直接启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash ./tiger_item_resolution_train.sh \
  --data-dir data/beauty --dataset beauty \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --arm mir --seed 42 --devices '[0,1]' --master-port 29810 \
  --notes "MIR v1; Beauty; seed 42; from scratch"
```

常规模型/训练 override 位于默认值之后；队列禁止用原生 override 改写 arm、seed、dataset、checkpoint 等矩阵身份，使用上述选择器或单条件入口。

## 6. 每个训练 run 完成后的单卡评价

下面为 Beauty/MIR/seed42 示例。把 `TRAIN_RUN_ID` 设置为本轮对应训练 run；其他条件同时替换 arm/seed，Sports 同时替换数据、SID 和 embedding 来源。不要把旧 CGBS checkpoint 传入新模型。

```bash
TRAIN_RUN_ID='填入本轮训练run-id'

NPROC_PER_NODE=1 bash ./tiger_item_resolution_inference.sh \
  --data-dir data/beauty --dataset beauty --data-split evaluation \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --checkpoint-path "wandb://${TRAIN_RUN_ID}" \
  --arm mir --seed 42 --gpu 0 \
  --notes "MIR v1 best checkpoint; evaluation; single GPU"
```

默认通过 `checkpoint` role 取 best。若统一补 last 的稳定性分析，使用 `"wandb://${TRAIN_RUN_ID}?role=checkpoint_last"`，它与 best 的 artifact role 分离，不产生选择歧义。模型会核对目录、arm、损失和初始化契约；训练时改了相关参数，评价必须显式传相同 override。

推理发布 `recommendation_output` 和 `item_resolution_trace` 两类产物。MIR/解析控制记录真实前沿事件、gate、到达与解析质量、状态/打分次数、剩余质量和 Top-K 证书，以及目标解析责任；其他控制记录候选/计算信息，不冒充 MIR 的概率界。

`model_batch_seconds` 测量 encoder + 搜索，`decoder_batch_seconds` 只测搜索，均含 GPU 同步，排除 trace 的额外 teacher-forcing 与写盘。每行保留 batch_size；比较时按 batch 分组、排除首个 batch 并同时报告吞吐，不能把同一个 batch 时间按用户重复相加，也不能由此宣称在线端到端 p99。

取得 inference run ID 后执行 outcome diagnosis：

```bash
INFERENCE_RUN_ID='填入对应inference-run-id'

bash ./tiger_item_resolution_diagnosis.sh \
  --data-dir data/beauty --data-split evaluation --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --recommendation-output-path "wandb://${INFERENCE_RUN_ID}" \
  --seed 42 --notes "MIR v1 keyed recommendation outcome; no legacy prefix trace"
```

新 item-resolution trace 不传给 `--fixed-prefix-trace-path`；旧 prefix survival 的“每层存活”与多出口模型语义不同。Outcome diagnosis 复用既有分组、Hit/Recall/NDCG 和逐用户证据，resolution 机制数据保留在独立 Artifact 中。

## 7. WIDE 风格适配控制

每个新 `hybrid` checkpoint 增加一次 training-only 标定和一次评价推理，不新增训练条件。以 Beauty/seed42 为例：

```bash
HYBRID_TRAIN_RUN_ID='填入本轮hybrid训练run-id'

NPROC_PER_NODE=1 bash ./tiger_item_resolution_calibration.sh \
  --data-dir data/beauty --dataset beauty \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --checkpoint-path "wandb://${HYBRID_TRAIN_RUN_ID}" --seed 42 --gpu 0 \
  --notes "WIDE-style entropy calibration; training split only"
```

标定固定读取 training 原始用户序列的最后训练目标；不使用训练时展开的全部子序列。这是与原方法标定采样的明确适配差异。记录每用户各层 teacher-forcing entropy，对训练用户求平均；产物携带 checkpoint 权重和目录内容指纹。

```bash
CALIBRATION_RUN_ID='填入上一步标定run-id'

NPROC_PER_NODE=1 bash ./tiger_item_resolution_inference.sh \
  --data-dir data/beauty --dataset beauty --data-split evaluation \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --checkpoint-path "wandb://${HYBRID_TRAIN_RUN_ID}" --arm hybrid --seed 42 --gpu 0 \
  --policy wide --calibration-path "wandb://${CALIBRATION_RUN_ID}" \
  --notes "WIDE-style adapted control; single GPU; training-only thresholds"
```

WIDE 默认上限为 4096 个路由状态和 4096 个最终 item 评分，batch 8；另有一个根 dense query 状态，trace 按实际数记录。wildcard 位置展开合法孩子且不加 log 惩罚，后续位置继续解码；最后按可靠位置概率与连续相似度混合。预算截断单独计数，不宣称其分数是归一化 item 概率或拥有 MIR 证书。跨方法延迟比较须显式统一 batch，如全部设置 `data.predict_dataloader.batch_size_per_device=8`。

## 8. 最近邻适配边界

- **COBRA**：保留 sparse→conditional dense 和 BeamFusion，使用既有 raw SID 三层、共享 TIGER encoder/decoder、冻结内容输入加可训练投影；dense 训练为全目录 CE，推断精确枚举当前 sparse 桶。未复现原文交替 sparse/dense 历史输入和可训练文本 encoder，名称应写 `COBRA-style matched adaptation`。它不能替代正式稿对官方实现/完整复现的外部基线义务。
- **WIDE**：将原跨模态 query 映射为 hybrid 的历史条件 query；使用已有四位 SID、训练用户末目标标定、显式有界 wildcard 扩张。它是方法机制控制，不能冒称原 M-BEIR 配置复现。
- **Hybrid**：本项目控制，不标为正式 LIGER 复现。

原始方法：[COBRA §3](https://arxiv.org/html/2503.02453)、[WIDE §4](https://arxiv.org/html/2609.03554)。本轮匹配对照用于决定 MIR 是否值得形成论文主张；若成立，正式投稿前仍需完成最近邻的复现完整性核查及迁移设置。

## 9. 预算曲线与研究决策

MIR/解析控制默认 Q=64、S=4096。在尚未形成 K 个候选时优先完成较深前沿，之后按到达质量展开；所有解析 arm 使用相同规则，仍受原 Q/S 上限约束。训练结果可用固定网格 Q∈{32,64,128}、S∈{1024,4096} 补单卡推理；通过 `search_states=32 search_item_scores=1024` 覆盖，不重训。任何预算点若无法产生 K 个正质量唯一 item 会明确失败，应作为预算不可行点记录，不补任意 item 或静默提高预算。

三 seed 汇总必须使用同一预算点；不能逐 arm、逐 seed 根据 testing 结果挑点。小目录 dense 若占优，承认适用范围；fixed depth 或 depth_gate 若解释全部收益，停止 MIR 自适应主张。只有完整比较满足门槛后，再决定 Toys/RVQ 迁移和论文最终题目。

## 10. 验证边界

本轮验证使用小型内存目录、CPU T5、Hydra compose/instantiate、共享 writer 往返、shell 命令预览及既有接口回归。没有执行真实训练、GPU inference、training calibration 或 diagnosis；没有发布新的实验 run。实际 GPU 内存、速度、收敛与效果由上述人工实验确认。
