## Why

当前 `configs/` 中剩余的 callback 相关文件里，有一类并非死配置，但本质上只是非常薄的包装层：例如只转发一个 progress bar 配置，或只为了 inference 再包一层 defaults。它们仍然增加了配置层级深度，却没有提供足够的复用价值。

在已经清理掉明显死配置之后，下一步适合先处理这类“薄包装配置”，以内联方式减少文件数量，同时保持 train / inference 当前行为不变。

## What Changes

- 以内联方式收敛薄包装 callback 配置，优先处理 `inference_default.yaml` 与 `one_based_tqdm_progress_bar.yaml`。
- 保持 train / inference 当前 callback 行为不变，不修改 experiment 内联 callback 定义。
- 不触碰仍承担默认模板职责的 `callbacks/default.yaml`、`model_checkpoint.yaml`、`early_stopping.yaml`、`model_summary.yaml`。

## Capabilities

### New Capabilities
- `thin-callback-config-inlining`: 将薄包装 callback 配置收敛到更直接的配置入口，减少不必要的文件跳转层。

### Modified Capabilities

## Impact

- 受影响文件预计包括：`configs/inference.yaml`、`configs/callbacks/inference_default.yaml`、`configs/callbacks/one_based_tqdm_progress_bar.yaml`、以及必要时 `configs/callbacks/default.yaml`
- 不涉及运行时逻辑实现，只调整 Hydra 配置组织方式
- 目标是减少配置文件数量并保持当前行为完全一致
