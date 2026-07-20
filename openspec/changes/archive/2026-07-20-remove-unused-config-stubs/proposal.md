## Why

当前 `configs/` 中存在几类已经不接入主配置链路的模板/占位配置，例如未被 defaults、experiment、脚本或文档引用的 callback/trainer 子配置，以及没有内容的本地配置目录。这些残留会增加认知噪音，让后续整理主配置层时更难分辨哪些文件仍然有效。

本次变更先做最小范围清理：只删除已确认未被引用的死配置与空本地目录，不触碰仍有潜在复用价值但当前重复度较高的配置层。

## What Changes

- 删除未被任何 train/inference defaults、experiment、脚本或文档引用的死配置文件。
- 删除当前仅包含 `.gitkeep` 且未被接入的 `configs/local/` 空目录占位。
- 保持 `train.yaml`、`inference.yaml`、`callbacks/default.yaml`、`logger/default.yaml`、`trainer/default.yaml` 等主配置层不变。
- 不在本次变更中重构 experiment 内联配置，不处理 OpenSpec 历史文档中的描述性内容。

## Capabilities

### New Capabilities
- `unused-config-pruning`: 清理未引用的配置模板与空本地配置目录，降低配置目录噪音。

### Modified Capabilities

## Impact

- 受影响文件预计包括：`configs/callbacks/local_pickle_writer.yaml`、`configs/callbacks/rich_progress_bar.yaml`、`configs/trainer/ddp.yaml`、`configs/local/.gitkeep`
- 不涉及运行时行为变更，只移除已经确认未接入的配置文件
- 为后续进一步精简 `configs/` 提供更清晰的基线
