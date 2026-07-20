## Why

刚完成的按组件拆分 experiment 配置虽然已经显著缩短了 `configs/experiment/*.yaml`，但拆分出的 `model/*`、`trainer/*`、`data_loading/*` 等文件内部仍然保留了 `components.*` 这一层包装，同时还保留了顶层参数域。这使组件文件依然带有多余层级，未完全达到“组件文件就是组件本身”的目标。现在需要继续收敛配置树，移除 `components` 容器层与独立参数域，只保留必要的 `root` 节点来表示顶层可实例化组件。

## What Changes

- **BREAKING** 移除配置树中的 `components` 容器层，Python 侧实例化入口改为直接读取顶层组件节点，例如 `cfg.data_loading.datamodule`、`cfg.model.root`、`cfg.trainer.root`、`cfg.callbacks`、`cfg.logger`。
- **BREAKING** 不再保留独立的顶层参数域（如旧的 `model:` / `trainer:` 纯参数视图）；组件参数直接在对应组件配置文件中维护。
- 让 `configs/model/*.yaml`、`configs/trainer/*.yaml`、`configs/data_loading/*.yaml`、`configs/logger/*.yaml`、`configs/callbacks/*.yaml` 通过更直接的 package 目标表达自身，不再在文件内部重复写 `model:` / `trainer:` / `data_loading:` / `logger:` / `callbacks:` 外层包装；挂载位置由 `configs/experiment/*.yaml` 的 defaults 显式指定。
- 保留 `root` 作为单实例顶层组件标识；集合型组件继续直接用 `callbacks` / `logger` 节点。

## Capabilities

### New Capabilities
- `rootless-components-container-removal`: official experiment configs SHALL expose top-level component entrypoints directly, without a `components` wrapper layer, while retaining `root` for single top-level components.

### Modified Capabilities

## Impact

- 受影响代码：`src/utils/launcher_utils.py` 及任何直接读取 `cfg.components` 的辅助代码
- 受影响配置：`configs/experiment/` 以及按组件拆分的 `configs/data_loading/`、`configs/model/`、`configs/trainer/`、`configs/logger/`、`configs/callbacks/`
- 主要收益：进一步降低组件配置层级复杂度，让组件文件内容与目录语义更一致
