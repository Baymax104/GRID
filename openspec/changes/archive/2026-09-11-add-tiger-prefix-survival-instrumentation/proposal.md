## Why

当前四组 TIGER baseline 只证明 Tail candidate accessibility 与 equal-risk outcome 较差，尚不能定位差距来自 teacher-forcing 概率、固定 beam 剪枝还是最终 SID-to-item 映射。DASFAA 主方法依赖这一机制判断，因此需要先提供 split-safe、可审计且不改变默认生成结果的逐层 prefix survival instrumentation，并用现有 checkpoint 完成 Go / No-Go pilot。

## What Changes

- 为 TIGER teacher-forcing 计算增加逐层 target token probability、rank、margin 等可选诊断输出。
- 为 constrained beam search 增加可选逐层 trace，记录 target prefix rank、survival、parent beam、累计分数与 first failure depth；instrumentation 关闭时保持现有生成结果和 recommendation bundle 不变。
- 将 trace 作为独立的 keyed Prefix Trace Artifact 写入和发布，不把异构诊断字段塞入现有 `merged_predictions_tensor.pt` recommendation bundle。
- 让 trace inference 显式选择 `evaluation` 或 `testing` split；用于方法统计与调参的 trace 必须来自 `evaluation`，避免 test leakage。
- 扩展 Tail-SID diagnosis，使其可选消费 key-aligned Prefix Trace Artifact，输出 layer-wise、risk-matched survival evidence 和 widened-beam recovery evidence。
- 暴露并验证 widened-beam inference override，使 beam 宽度可在复用 checkpoint 时改变而无需重新训练。
- 增加 decoder 级行为保持、合法 prefix、beam parent、target rank/survival、Artifact 对齐与 split 隔离测试。
- 固化四个现有 seed-42 checkpoint 的机制 pilot 方案与 Go / No-Go 判据；本变更不实现 Budget-Aware Prefix Calibration。

## Capabilities

### New Capabilities

- `tiger-prefix-survival-tracing`: 定义 TIGER teacher-forcing、fixed-beam 与 widened-beam 的逐层目标路径 trace、行为保持和 split-safe 运行契约。
- `tiger-prefix-trace-artifact`: 定义独立、keyed、可本地使用并可发布到 W&B 的 Prefix Trace Artifact 协议。

### Modified Capabilities

- `self-contained-tiger-generation-model`: 扩展自包含 TIGER decoder，使其在不改变默认 generation 行为的前提下可选返回逐层 beam trace。
- `tiger-sequence-data-contract`: 允许 trace inference 保留 `target_ids`，并要求显式选择 evaluation/testing 数据 split，同时继续把 user identity 仅作为输出 key。
- `prediction-output-protocol`: 允许运行时 `ModelOutput` 携带不进入标准 recommendation bundle 的可选 auxiliary tensor payload，供独立 trace writer 消费。
- `tail-sid-diagnosis-evidence-artifact`: 允许 diagnosis 消费 Prefix Trace Artifact，并输出稳定的逐层 survival、first-failure 与 widened-beam recovery 证据文件。

## Impact

- 主要影响 `src/recommendation/tiger/decoder.py`、`src/recommendation/tiger/tiger.py`、TIGER batch/output data models、trace writer、Artifact resolver、Tail-SID diagnosis data/evidence，以及对应 Hydra experiment/data/model/callback 配置。
- 新增 decoder 与 trace Artifact 聚焦测试；现有 recommendation output、metric callback、统一 `src.main`、launcher、W&B logger 生命周期和 Artifact lineage ownership 保持不变。
- 现有四个 finished TIGER checkpoint 与 Semantic ID Artifact 可直接复用；pilot 只增加 inference/analysis，不增加训练任务或新依赖。
