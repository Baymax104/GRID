## Why

上一轮 experiment 配置组件化之后，主要装配子树已经被提升到 `components:`，但 Python 侧顶层实例化入口仍然部分依赖参数域别名，如 `cfg.data_loading.datamodule` 仍转发到 `components.data_loading.datamodule`。这使依赖关系仍然存在一层不必要的中转，未完全达成“参数域只承载参数、组件域才承载装配根”的目标。

现在需要把 Python 侧中直接用于实例化的配置引用统一收敛到 `components`，使依赖关系规范化为 `python -> components config -> argument config`，从而进一步厘清参数域与装配域的职责边界。

## What Changes

- **BREAKING** 将 Python 侧对 datamodule、model、callbacks、logger、trainer 的直接实例化入口从参数域迁移到 `components`。
- 为 official experiment 配置补齐统一的组件入口命名：
  - `components.data_loading.datamodule`
  - `components.model.root`
  - `components.trainer.root`
  - `components.callbacks`
  - `components.logger`
- 收缩 `data_loading`、`model`、`trainer` 等参数域，使其不再承担 Python 直接实例化入口别名。
- 允许日志与超参数记录继续优先保留参数域语义，必要时再补充组件域信息。

## Capabilities

### New Capabilities
- `python-component-entrypoint-normalization`: Python 侧顶层实例化逻辑直接依赖 `components` 下的装配根，而不是经由参数域中转。

### Modified Capabilities

## Impact

- 受影响代码：`src/utils/launcher_utils.py` 及必要的日志/辅助代码
- 受影响配置：所有 official `configs/experiment/*.yaml`
- 不引入 repo 级共享配置，也不改变 Hydra 顶层装配机制，但会重排配置引用关系
