## Why

当前仓库保留了一套 restart / retry 机制，但该机制已经不在默认训练入口中启用。尽管如此，主链路中的 `launcher_utils.py`、训练脚本参数以及若干注释仍然保留了对 restart 机制的感知，增加了理解成本，也让默认运行路径显得不够干净。

现在需要将 restart 机制从默认主链路中去耦，同时保留历史代码供参考或手动接入，并通过轻量注释明确其已废弃状态。

## What Changes

- 从默认 train / inference 主链路中移除对 restart metadata 的依赖。
- 保留 `ckpt_path` 为目录时自动解析最新 checkpoint 的能力，不把该能力视为 restart 机制的一部分。
- 清理默认训练脚本中与 restart 机制相关的遗留参数。
- 为 `src/utils/restart_job.py` 和 `src/utils/restart_job_utils.py` 增加轻量废弃标注，说明其不再属于默认主链路。
- 不删除 restart 代码文件，不修改依赖清单。

## Capabilities

### New Capabilities
- `mainline-restart-decoupling`: 规定默认训练/推理主链路不得再依赖 restart 机制，同时保留显式按目录选择最新 checkpoint 的能力，并将 restart 模块标注为废弃。

### Modified Capabilities

## Impact

- 受影响代码：`src/utils/launcher_utils.py`、`src/train.py`、训练 shell 脚本、`src/utils/restart_job.py`、`src/utils/restart_job_utils.py`
- 不涉及新增外部依赖，不涉及 checkpoint 文件格式变更
- 默认运行路径的心智负担会降低，restart 代码转为边缘/历史能力
