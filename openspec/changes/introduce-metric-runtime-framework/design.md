## Context

当前 GRID 训练指标由模型直接管理：

- TIGER 通过配置注入 `SIDRetrievalEvaluator`，但仍把每个指标 `setattr` 到 `LightningModule`。
- RKMeans、RVQ、RQVAE 在构造函数中声明大量 `MeanMetric`，并根据 `n_layers` 动态创建 per-layer 指标。
- 各模型在 `training_step`、validation/test hook 中直接 update/reset/log 指标。

用户要求指标框架与 `LightningModule` 低耦合，不要在各模型中重复侵入 hook。因此指标生命周期应放到 Trainer callback 侧，模型只返回可评价 payload。

## Goals / Non-Goals

**Goals:**

- 提供一个配置驱动的 torchmetrics 运行时框架。
- 指标定义可由 model 配置声明，并支持 train/val/test stage 隔离。
- 用单一 `MetricCallback` 统一接入 Lightning hooks。
- 支持根据配置值动态展开指标，例如 quantization 的 per-layer coverage/entropy。
- 支持普通 scalar 指标和自定义指标组。
- 提供不运行完整 experiment 的单元测试。

**Non-Goals:**

- 本变更不迁移 TIGER 或 quantization 模型。
- 本变更不改变现有指标计算公式。
- 本变更不新增外部依赖。
- 本变更不要求指标 state 写入训练 checkpoint。

## Decisions

### 指标生命周期放在 Callback，而不是 LightningModule

`MetricCallback` 持有 `MetricEngine`，在 batch end hook 中从 step output 提取 payload 并更新指标，在 epoch end hook 中统一 log/reset。

理由：

- 模型不需要声明指标属性或实现指标 hook。
- 所有模型共享一套指标生命周期。
- 后续迁移只需要让 step 返回指标 payload。

替代方案是在每个模型持有 `MetricEngine` 并调用 `self.metrics.update/log`。该方案仍会让模型重复接触 hook 和日志策略，不满足低耦合要求。

### MetricEngine 使用 ModuleDict 注册指标

`MetricEngine` 继承 `torch.nn.Module`，内部按 stage 持有 `ModuleDict`。`MetricCallback.setup` 将 engine 移到 `pl_module.device`。

理由：

- torchmetrics 本身是 `nn.Module`，需要参与 device 迁移。
- 单一 engine 比大量模型属性更容易测试和迁移。

### 指标输入通过 resolver 配置声明

每个指标声明 `input`，默认从 callback 提供的 payload 中按 key 取值，也支持按 index 取列表/张量元素。

理由：

- 普通 `MeanMetric` 和动态 layer 指标可以共用同一机制。
- 复杂领域转换可放到自定义 metric group 内部，避免把转换逻辑扩散到模型。

### 动态指标通过 repeat 展开

配置支持：

- `count`
- `name_template`
- `input.index`
- 可选 `index_name`

engine 初始化时把 repeat 规范展开为多个真实指标。

理由：

- quantization 的 per-layer 指标数量由 `num_hierarchies` 决定。
- 指标名称和输入位置都应由配置控制，不能回到动态 `setattr`。

## Risks / Trade-offs

- [Risk] Callback 持有的指标默认不随模型 checkpoint 保存。  
  Mitigation: 训练指标通常是运行时聚合状态，第一版不恢复；未来若需要可为 `MetricCallback` 添加 `state_dict/load_state_dict`。

- [Risk] 模型 step output 结构成为指标框架的隐式契约。  
  Mitigation: 后续模型迁移的 OpenSpec 必须明确每个模型返回的 payload 字段，并用单元测试覆盖。

- [Risk] 复杂指标输入转换可能让通用 engine 膨胀。  
  Mitigation: engine 只处理通用生命周期和简单 input resolve；复杂领域指标放到 metric group 中。

## Migration Plan

1. 先引入框架与单元测试，不改现有模型行为。
2. 独立变更迁移 TIGER，移除 `Evaluator` 注入与模型内指标 hook。
3. 独立变更迁移 quantization 模型，使用 repeat 指标覆盖 per-layer 指标。
4. 搜索残留 `MeanMetric`、`Evaluator`、模型内指标 hook，清理兼容代码。
