## Why

Beauty seed42 的 BRIR 三个分支没有超过冻结 base，dense 分支在候选 CE 下降时全目录指标严重退化。现有日志不能区分目标支持集差异、评分退化与搜索遗漏，需要一次有界诊断收尾，并补齐 Sports 内容初始化对照。

## What Changes

- 为现有 BRIR audit 增加可选训练诊断：保留全目录分数、冻结候选来源、目标排名、两种 CE、候选外竞争及残差饱和证据。
- 增加 base、dense、prefix_free、brir 共用 128 个 evaluation 用户的手动执行套件，固定既有 last checkpoint 和上游来源。
- 验证零更新分支初始化一致性，记录零残差反事实的含义，避免将其当作新训练实验。
- 核对 Sports `token_content_init` 的既有 20k、有效 batch256、GPU 0/1 训练命令。

## Capabilities

### New Capabilities

- `brir-training-diagnostics`: 冻结 checkpoint 的训练目标与全目录排序诊断、可复核证据和有界手动入口。

### Modified Capabilities

无。原有 audit v1 和训练协议继续可用。

## Impact

涉及 BRIR audit、reference 搜索结果、证据校验、Hydra 配置、根目录套件与聚焦测试。不改变训练目标、模型参数、checkpoint 契约或推理决策，不增加依赖。完整 GPU 任务由用户启动。
