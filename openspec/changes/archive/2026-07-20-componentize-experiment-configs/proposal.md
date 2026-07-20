## Why

当前 `configs/experiment/*.yaml` 同时混合了两类内容：一类是直接或递归用于 `hydra.utils.instantiate` 的组件装配子树，另一类是实验本身的纯参数/共享参数。这两类内容交错定义，尤其在 `data_loading` 与 `model` 区域里层次很深、嵌套很多，使主链路难以快速看清“这个实验到底由哪些主要组件构成”。

现在需要对所有 official experiment 配置做单文件内组件化重组：在不引入 repo 级共享配置、也不改变 Python 装配入口的前提下，把主要组件提升到统一的 `components:` 区，把 `data_loading`、`model` 等收敛为参数域，从而显著提高 experiment 文件的可读性和新增实验时的可维护性。

## What Changes

- **BREAKING** 重组所有 official experiment 配置文件的内部骨架，统一引入 `components:` 区表达主要 instantiate 子树。
- 将 `data_loading` 与 `model` 收敛为参数域名称，主要承载纯参数、共享参数和组件引用，而不是继续内联主要装配子树。
- 将 experiment 中主要的 datamodule、dataloader、dataset、collate、label、quantization 子模块、推荐模型子模块等组件提升为具名节点。
- 允许保留少量强局部、一次性、短小的 inline `_target_`，避免过度抽象。
- 保持 `src/utils/launcher_utils.py` 的顶层装配入口不变，确保行为重点落在配置组织而非 Python 运行时逻辑改造上。

## Capabilities

### New Capabilities
- `experiment-config-componentization`: official experiment 配置在单文件内显式区分主要组件与参数域，使主链路更清晰、实验定义更易浏览和维护。

### Modified Capabilities

## Impact

- 受影响配置：`configs/experiment/*.yaml` 全部 official experiment 文件
- 可能受影响的默认入口：如有必要仅做最小适配，但本次以 experiment 改动为主
- 不引入 repo 级共享配置目录，不做跨 experiment 配置复用
- 不直接改动 Python launcher 逻辑，但会显著改变 experiment YAML 的内部组织方式
