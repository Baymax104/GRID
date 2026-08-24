## Why

W&B project/group 目前分散在 experiment、logger、callback、data 和 model 配置中，且同时存在 `wandb_project`、硬编码 `GRID`、`${task_name}` 等写法。将 project/group 收敛到 experiment 顶层可以让实验归属一眼可见，并避免组件配置重复定义实验链路语义。

## What Changes

- 删除 official experiment 配置中的 `wandb_project` 顶层字段。
- 在 official experiment 配置顶层统一声明 `project` 与 `group`。
- `group` 使用实验族名，例如 `rqvae`、`rvq`、`tiger`，不使用 `${task_name}`。
- W&B logger、artifact writer、checkpoint writer 的 `project` / `group` 统一引用 `${project}` / `${group}`。
- data/model 中调用 W&B artifact 读取函数时，保留函数参数名 `wandb_project`，但配置值统一引用 `${project}`。
- `src.main` 的短 W&B URI 默认 project 解析改为读取顶层 `project`。

## Capabilities

### New Capabilities

### Modified Capabilities
- `experiment-config-componentization`: Official experiment configs own W&B project/group identity, while component configs reference those top-level fields.

## Impact

- Affected configs:
  - `configs/experiment/*.yaml`
  - `configs/logger/*.yaml`
  - `configs/callbacks/*.yaml`
  - `configs/data/*.yaml`
  - `configs/model/tiger_*.yaml`
- Affected code:
  - `src/main.py`
- Tests should verify `cfg.project` is passed as the default W&B project for short artifact/checkpoint URIs.
