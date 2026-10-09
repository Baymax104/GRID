## Why

用户接受现有约 8% 的预算匹配收益，要求把依赖 50k 后续训及双 checkpoint 部署的方法改为可从随机初始化整体训练的模型，每个最终模型最多 50000 更新。当前正向共享协同残差可以直接从零初始化参与 v0 联合目标，但其 scratch 效果尚未验证；旧续训与 pooling 结果不能代替本目标的证据。

## What Changes

- 新增单模型 CoPMRec unified-50k：v0 三项联合损失、learned alpha、共享历史与目录的 seen 商品残差，以及与 LIGER 完全相同的输入历史资格排除。
- 固定一次 0→50000 连续训练，主干 peak LR 0.0003、残差 peak LR 0.002、warmup2500/cosine50000；禁止外部预训练、teacher、阶段 optimizer 重建和 checkpoint pooling。
- 提供统一 src.main 的双卡训练与单卡推理薄配置、根脚本、严格 checkpoint/训练预算契约及聚焦测试。
- 以原 50k LIGER 单模型的同 history dense 输出为 seed42 主对照；若首个 scratch 验证达到双指标至少 8%，冻结方法并执行 seed43 的原生 LIGER／相同 CoPMRec 成对复现。
- 新阶段显式累计上限 3 正式训练／150000 新更新；每个模型最多50000。完整 Validation 最多3次，新 Testing最多3次且仅用于冻结模型的验收；旧所有阶段额度保持封存，不重置。

## Capabilities

### New Capabilities

- `copmrec-unified-50k`: 单次预算内整体训练、单成员部署、来源和训练契约、对匹配 LIGER 的效果及复现审计。

### Modified Capabilities

无。保留 v0、v4、旧 scratch-ranking 和预算匹配续训的行为及历史证据。

## Impact

新增 recommendation 薄派生模块、model/trainer/experiment 配置及根 shell 入口；复用现有 v0 联合概率、共享残差、history exclusion、data、writers、source snapshot 和 launcher。无需新依赖。GRID 记录执行与证据，research-state/current-plan 记录新增授权、累计成本与当前判断；不改变旧报告或产物。
