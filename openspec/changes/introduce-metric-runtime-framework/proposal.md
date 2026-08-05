## Why

当前训练指标分散在各个 `LightningModule` 中：模型直接持有 `MeanMetric`、动态 `setattr` 指标、手写 `self.log_dict`，并重复实现 validation/test hook。随着 TIGER 与 quantization 模型的指标种类增加，这会让模型训练逻辑、指标生命周期和日志策略持续耦合。

本变更引入一个配置驱动的指标运行时框架，让指标定义留在 model 配置中，但指标更新、计算、日志和重置由统一 callback 承担。

## What Changes

- 新增 `src/common/metrics/` 指标运行时模块。
- 新增 `MetricEngine`，负责按 stage 管理 torchmetrics 指标、更新、计算、重置和日志。
- 新增 `MetricCallback`，统一接入 Lightning train/validation/test batch 与 epoch hooks。
- 新增 payload extractor / input resolver，用于从 step output、batch 或 `pl_module` 中提取指标输入。
- 支持配置驱动的动态指标展开，例如按 `${num_hierarchies}` 生成 per-layer quantization 指标。
- 保留现有模型行为不变；本变更只提供框架和测试，不迁移具体模型。

## Capabilities

### New Capabilities
- `metric-runtime-framework`: 配置驱动的 torchmetrics 运行时框架，负责低耦合地管理指标生命周期、日志和动态指标展开。

### Modified Capabilities

## Impact

- 影响新增代码：`src/common/metrics/`。
- 影响测试：新增 `tests/common/metrics/` 单元测试。
- 后续迁移会影响 `src/recommendation/tiger/tiger.py`、`src/quantization/*` 和 `configs/model/*_train.yaml`，但不在本变更内完成。
