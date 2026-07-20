## Why

当前项目缺少一个低成本的“链路冒烟”运行模式。开发者若想验证 train / inference 主链路、数据加载、模型装配和 Lightning 运行是否正常，通常会触发 checkpoint、预测结果和实验日志等业务产物写入，既增加成本，也污染输出目录。

现在需要提供一个通过命令行参数启用的 dry run 模式，让用户能以最小运行规模验证主链路，同时避免写入业务结果。

## What Changes

- 新增命令行参数 `--dry-run`，供 `src.train` 与 `src.inference` 入口启用 dry run。
- 将 `--dry-run` 转换为统一的内部配置开关，例如 `dry_run=true`，避免与 Hydra CLI 冲突。
- 在 dry run 下保留真实主链路执行，但将运行规模压缩为单 batch / 单 step smoke。
- 在 dry run 下禁止写入业务结果，包括 checkpoint、prediction pickle/tensor、CSV logger、W&B logger。
- 保留 Hydra 输出目录、普通日志与 `config_tree.log` 等运行元信息写入。

## Capabilities

### New Capabilities
- `cli-dry-run`: 通过 `--dry-run` 启用最小规模的 train / inference 冒烟运行，并禁止业务结果写入。

### Modified Capabilities

## Impact

- 受影响代码：`src/train.py`、`src/inference.py`、`src/utils/launcher_utils.py`、`src/utils/instantiators.py`、推理写入 callback 相关代码及默认配置装配路径
- 不要求修改依赖清单
- 默认非 dry run 行为保持不变
