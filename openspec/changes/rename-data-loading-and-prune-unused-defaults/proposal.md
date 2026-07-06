## Why

当前 official 入口已经不再依赖 `configs/trainer/default.yaml`、`configs/logger/default.yaml`、`configs/callbacks/default.yaml`，这些默认配置及其关联模板层很可能已经成为死配置。同时，当前配置树仍残留旧命名 `data_loading` 的历史痕迹，而新的扁平化组件结构更适合统一使用更简洁的 `data`。现在需要清理无用默认配置，并完成 `data_loading -> data` 更名，以减少噪音并统一配置语义。

## What Changes

- **BREAKING** 将官方配置与 Python 读取路径中的 `data_loading` 统一更名为 `data`。
- 删除不再被 official experiment 入口使用的 `trainer/default.yaml`、`logger/default.yaml`、`callbacks/default.yaml`；并评估/清理随之失去引用的 callback 模板文件。
- 更新 split config 目录与 experiment defaults 挂载路径，使 `configs/data_loading/` 迁移为 `configs/data/`，并同步修正所有跨配置插值。
- 同步修正日志记录、config tree 打印、配置缺失 warning 等仍使用 `data_loading` 旧名称的辅助代码。

## Capabilities

### New Capabilities
- `data-config-rename-and-default-pruning`: official experiment configs SHALL use `data` as the top-level data component name and SHALL not retain unused default component config stubs.

### Modified Capabilities

## Impact

- 受影响目录：`configs/data_loading/`（改名为 `configs/data/`）、`configs/experiment/`、`configs/trainer/`、`configs/logger/`、`configs/callbacks/`
- 受影响代码：`src/utils/launcher_utils.py`、`src/utils/logging_utils.py`、`src/utils/utils.py`、`src/utils/rich_utils.py` 等直接使用 `data_loading` 名称的代码
- 受影响文档/spec：当前活跃 change 与 living spec 中对 `data_loading` / 旧 default 配置的描述
