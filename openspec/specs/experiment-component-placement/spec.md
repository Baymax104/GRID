# experiment-component-placement Specification

## Purpose
TBD - created by archiving change move-prediction-writers-to-common. Update Purpose after archive.
## Requirements
### Requirement: 实验组件 SHALL 放在职责匹配的 src 域目录中
被 Hydra `_target_` 实例化的 Lightning callbacks 和其他实验组件 SHALL 放在职责匹配的 `src` 域目录中，而非 `src/utils/`。`src/utils/` 仅保留通用工具（日志、文件 I/O、弹性装饰器等），不承载被 Hydra 实例化的实验组件。

#### Scenario: prediction writer 归属 common inference 域
- **WHEN** 维护者检查 `BaseBufferedWriter` 或 `LocalPickleWriter` 的位置
- **THEN** 它们 MUST 位于 `src/common/inference/prediction_writers.py`
- **THEN** 它们 MUST NOT 位于 `src/utils/`

#### Scenario: 新增实验组件时选择正确目录
- **WHEN** 维护者需要新增一个被 Hydra `_target_` 实例化的 Lightning callback 或实验组件
- **THEN** 该组件 MUST 放在职责匹配的 `src` 域目录中
- **THEN** 该组件 MUST NOT 放在 `src/utils/` 中
