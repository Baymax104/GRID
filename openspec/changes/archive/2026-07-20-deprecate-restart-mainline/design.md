## Context

当前默认训练/推理链路由 `src/train.py`、`src/inference.py` 和 `src/utils/launcher_utils.py` 组成。虽然外层 `LocalJobLauncher` 与 `RestartAndLoadCheckpointCallback` 已经不在默认入口中启用，但 `launcher_utils.py` 仍会在初始化阶段导入 `restart_job_utils` 并读取 `restart_metadata.json`，训练 shell 脚本中也保留了 `+should_skip_retry=true` 这类遗留参数。

与此同时，`update_cfg_with_most_recent_checkpoint_path()` 还承担另一个独立能力：当 `ckpt_path` 指向目录且 `should_retrieve_latest_ckpt_path=true` 时，自动选择最新 checkpoint。该能力对主链路仍有价值，不应随着 restart 机制一起移除。

## Goals / Non-Goals

**Goals:**
- 让默认 train / inference 主链路不再依赖 restart metadata、restart callback 或 launcher 语义。
- 保留按目录自动解析最新 checkpoint 的现有能力。
- 保留 restart 相关代码文件，但通过轻量注释明确其为废弃历史能力。
- 移除默认 shell 命令中无意义的 restart 参数遗留。

**Non-Goals:**
- 不删除 `src/utils/restart_job.py` 或 `src/utils/restart_job_utils.py`。
- 不重写 checkpoint 恢复机制。
- 不修改依赖清单、Hydra 总体装配方式或 Lightning 主流程。
- 不清理与 restart 无关的 dataloader retry 逻辑。

## Decisions

### 1. 将 restart metadata 恢复分支从 `launcher_utils.py` 移除
- 决策：`update_cfg_with_most_recent_checkpoint_path()` 仅保留“目录解析最新 checkpoint”分支，删除基于 `restart_metadata.json` 的自动恢复逻辑。
- 原因：这是主链路当前唯一真实的 restart 耦合点，去掉后 `train.py` / `inference.py` / `pipeline_launcher()` 都无需感知 restart。
- 备选方案：保留该逻辑但以 config 开关包裹。未采用，因为默认主链路仍会持有 restart import 和额外认知负担。

### 2. 保留目录选最新 checkpoint 的独立能力
- 决策：继续保留 `ckpt_path` 为目录时自动选择最近 checkpoint 的逻辑，并保留 `should_retrieve_latest_ckpt_path`。
- 原因：该能力可独立服务推理和手动恢复，不应被误归类为 restart 机制。
- 备选方案：一并删除全部自动解析逻辑。未采用，因为会损失现有有用能力。

### 3. restart 模块仅做轻量废弃标注
- 决策：在 `src/utils/restart_job.py` 与 `src/utils/restart_job_utils.py` 文件头或核心类 docstring 中标注“deprecated / 不属于默认主链路”。
- 原因：满足保留历史代码的要求，同时给后续维护者明确预期。
- 备选方案：移动到归档目录或单独文档说明。未采用，因为会扩大改动范围。

### 4. 清理默认脚本里的 restart 参数遗留
- 决策：从当前默认训练 shell 脚本中移除 `+should_skip_retry=true`。
- 原因：主链路不再使用 `LocalJobLauncher`，该参数没有实际效果，只会制造误导。
- 备选方案：保留参数并加注释说明无效。未采用，因为仍会污染默认命令模板。

## Risks / Trade-offs

- [历史用户依赖 restart metadata 自动恢复] → 通过保留 restart 代码文件和废弃注释降低断层，但默认主链路将不再提供该行为。
- [有人把 `should_skip_retry` 误认为仍然有效] → 删除默认脚本参数并在 restart 模块上标注废弃。
- [目录解析最新 checkpoint 与 restart 能力概念混淆] → 在实现和注释中明确该逻辑属于通用 checkpoint 解析，不依赖 restart metadata。

## Migration Plan

1. 先从 `launcher_utils.py` 去掉 restart import 和 metadata 读取逻辑。
2. 更新默认训练 shell 脚本，删除 `+should_skip_retry=true`。
3. 为 restart 模块添加废弃注释。
4. 做最小 smoke 检查，确认 train/inference 配置路径仍可解析。

回滚策略：若需要恢复旧行为，可单独恢复 `launcher_utils.py` 中的 metadata 分支，restart 模块文件仍保留在仓库中。

## Open Questions

- 当前无阻塞性开放问题。
