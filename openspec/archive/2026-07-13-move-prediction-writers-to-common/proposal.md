## Why

`src/utils/inference_utils.py` 包含 `BaseBufferedWriter` 和 `LocalPickleWriter` 两个 Lightning callback，是推理结果写入的实验组件。它们已依赖 `src/common/components/model_output.ModelOutput`，且与 `src/common/components/` 中的 `eval_metrics`、`loss_functions`、`scheduler` 同类——都是被 Hydra 配置实例化的实验组件。当前放在 utils 中不符合域归属。

## What Changes

- 将 `src/utils/inference_utils.py` 移至 `src/common/components/prediction_writers.py`（重命名以匹配 components 目录的功能描述性命名风格）
- 更新 4 处 Hydra `_target_` 字符串引用：
  - `src/utils/launcher_utils.py` 中 `DRY_RUN_DISABLED_CALLBACK_TARGETS` 的 1 处
  - `configs/callbacks/{sem_embeds_inference, rkmeans_inference, tiger_inference}.yaml` 各 1 处

## Capabilities

### New Capabilities
- `experiment-component-placement`: 被 Hydra `_target_` 实例化的 Lightning callbacks 和实验组件 SHALL 放在 `src/common/components/` 中，而非 `src/utils/`

### Modified Capabilities
<!-- 无。不涉及现有 spec 的行为变更。 -->

## Impact

- **1 个文件移动 + 重命名**：`src/utils/inference_utils.py` → `src/common/components/prediction_writers.py`
- **1 处 Python 字符串更新**：`launcher_utils.py` 的 `DRY_RUN_DISABLED_CALLBACK_TARGETS`
- **3 处 YAML 配置更新**：`configs/callbacks/` 下 3 个 inference callback 配置的 `_target_`
- **无 Python import 变更**：`inference_utils.py` 不被任何 Python 文件直接 import，仅通过 Hydra `_target_` 字符串实例化
- **无行为变更**、**无 API 变更**、**无 checkpoint 影响**
