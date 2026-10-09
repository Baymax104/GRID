## Why

MIR 的首 seed 及配对评分审计未支持继续投入潜在解析深度。用户已采纳 BRIR 设计：检验固定 SID 前缀对 dense 排序的独立修正价值，并测量有界分数精算的成本。

## What Changes

- 新增独立 BRIR 模型，包含基础 dense、dense 继续训练、无前缀残差及 SID 前缀残差。
- 建立共享基础 checkpoint、训练历史标定及冻结候选来源契约。
- 提供全目录 reference、动态上界、固定候选检索及可验证证据。
- 增加 Hydra 配置、手动运行脚本及内存测试；完整 GPU 实验由用户启动。

## Capabilities

### New Capabilities

- `bounded-residual-item-retrieval`: BRIR 训练、标定、检索、产物和手动实验入口。

### Modified Capabilities

无。

## Impact

新增 `src/recommendation/brir/`、data helper 与配置/根脚本；复用 TigerEncoder、MetricEngine、共享 writer 和 Artifact lineage。旧 MIR/CGBS 评分及 checkpoint 不变。不新增依赖、不启动训练、不提交 Git。
