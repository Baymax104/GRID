## Context

当前项目中的训练模块对 progress bar 的使用不一致。部分模块只在终端显示 `loss`，但 `ResidualQuantization` 会把量化损失、重构损失、残差范数、centroid 范数、覆盖率、熵等大量指标通过 `log_dict(..., prog_bar=True)` 直接输出到终端，导致训练日志冗长且影响核心信号识别。当前需求进一步收紧为：控制台不显示任何 metric，只保留进度条。

用户希望统一项目中的训练日志策略：终端不显示 `train/loss`、`val/loss`、`test/loss` 在内的任何数值指标，而其他指标保留写入 logger，不从 progress bar 暴露。

## Goals / Non-Goals

**Goals:**
- 统一训练、验证、测试阶段的终端日志策略，progress bar 不显示任何 metric。
- 保留其余指标的计算与 logger 输出，不删除指标本身。
- 以最小修改方式调整现有 `self.log` / `self.log_dict` 调用。

**Non-Goals:**
- 不修改 logger 后端（CSV/W&B）的整体策略。
- 不删除 verbose 指标，也不改变其语义。
- 不修改 dry run 逻辑或训练配置含义。

## Decisions

### 1. 终端可见性与 logger 输出分离
- 决策：训练、验证、测试阶段所有 `self.log` / `self.log_dict` 调用统一改为 `prog_bar=False`，并继续保留 `logger=True`。
- 原因：这能同时满足“控制台只保留进度”和“指标不丢”的目标。
- 备选方案：直接删除非 loss 指标日志。未采用，因为会损失实验分析信息。

### 2. 优先在模块内部最小化调整 `prog_bar`
- 决策：对于像 `ResidualQuantization` 这种当前把所有指标打包到 `log_dict(..., prog_bar=True)` 的模块，保留指标结构与 logger 参数，只将 `prog_bar` 统一改为 `False`。
- 原因：改动范围最小，且最直接满足需求。
- 备选方案：通过全局 logger hook 过滤 progress bar 字段。未采用，因为 Lightning 默认日志调用分散在模型内部，更难稳定拦截。

- ### 3. 对公共训练模块统一同一策略
- 决策：检查公共训练模块（如 `TransformerBaseModule` 等）中 train/val/test 阶段的 `self.log` 调用，统一关闭 progress bar metric 展示。
- 原因：避免项目内不同模型日志体验不一致。

## Risks / Trade-offs

- [某些开发者依赖终端观察 loss] → 指标仍保留在 logger 中，必要时后续可增加显式配置开关。
- [遗漏某些模块导致行为不一致] → 实施时搜索全部 `prog_bar=True` 的训练相关调用并逐一审视。
- [拆分 log 调用后行为偏差] → 保持 logger/on_step/on_epoch/sync_dist 参数不变，仅调整 progress bar 可见性。

## Migration Plan

1. 搜索训练/验证/测试阶段的 `self.log` / `self.log_dict` 调用。
2. 优先修改 `ResidualQuantization`。
3. 修改公共训练模块，统一关闭 metric 的 progress bar 展示。
4. 验证终端不显示任何 metric，而 logger 仍能接收指标。

## Open Questions

- 当前无阻塞性开放问题。
