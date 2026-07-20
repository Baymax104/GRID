## Why

当前 official experiment 配置把大部分组件参数和装配关系揉在同一个 `configs/experiment/*.yaml` 文件中，导致文件过长，并且许多构造参数需要同时在参数域和实例化域维护。现在需要按组件类型拆分 experiment 配置，让 experiment 文件回到“总装配入口”的角色，同时减少改参数时的重复修改点。

## What Changes

- **BREAKING** 将 official experiment 的大部分组件配置从 `configs/experiment/*.yaml` 下沉到按组件类型划分的配置组目录，例如 `configs/trainer/`、`configs/model/`、`configs/data_loading/`、`configs/logger/`、`configs/callbacks/`。
- `configs/experiment/*.yaml` 调整为薄入口：主要负责 defaults 导入、实验元信息、少量顶层输入，以及必要的跨组件装配关系。
- 各组件配置文件按实验名一一对应，不追求跨实验复用；单个实验的 trainer/model/logger/callbacks/data_loading 参数只在对应组件文件中维护一份。
- 保留 Python 侧从 `cfg.components` 读取实例化入口的规范，但让这些入口尽量由组件配置文件直接提供，而不是在 experiment 文件里手工转发大量参数。

## Capabilities

### New Capabilities
- `component-grouped-experiment-configs`: official experiments SHALL be composed from per-component config groups while preserving explicit dependency wiring.

### Modified Capabilities

## Impact

- 受影响目录：`configs/experiment/` 以及新增/扩展的 `configs/trainer/`、`configs/model/`、`configs/data_loading/`、`configs/logger/`、`configs/callbacks/`
- 受影响代码：可能需要少量更新配置说明文档与注释；Python 装配入口预期不大改
- 主要收益：缩短 experiment 主文件、减少双处维护、让组件参数修改集中到单一文件
