## Why

当前项目中部分训练模块会把大量训练指标直接输出到 progress bar / 终端，导致单轮训练日志冗长、难以快速定位核心信号。当前需求进一步收紧为：控制台不打印任何 metric，只保留训练进度本身，而所有指标继续保留给 logger 使用。

现在需要统一项目中的训练日志策略：终端不显示任何 `train/*`、`val/*`、`test/*` 数值指标，只保留进度条，而所有指标继续保留给 logger 使用。

## What Changes

- 统一训练相关模块的 progress bar 日志策略，终端不显示任何训练/验证/测试指标。
- 将训练/验证/测试指标统一改为 `prog_bar=False`，但继续保留 `logger=True`。
- 优先以最小修改方式调整现有 `self.log(...)` / `self.log_dict(...)` 调用，而不是删除指标本身。
- 覆盖当前已识别的训练模块，尤其是 `ResidualQuantization` 这类会输出大量 verbose metrics 的模块。

## Capabilities

### New Capabilities
- `training-log-verbosity-control`: 统一控制训练阶段终端日志可见性，不展示任何 metric，只保留进度，同时保留指标写入 logger。

### Modified Capabilities

## Impact

- 受影响代码：训练/验证/测试阶段使用 `self.log` 或 `self.log_dict` 的模型模块，重点包括 `src/models/quantization/residual_quantization.py` 和公共训练模块
- 不涉及依赖变更
- 不改变指标计算本身，只改变终端可见性与 progress bar 输出策略
