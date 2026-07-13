## Why

`src/utils/` 累积了约 816 行死代码：2 个整文件已废弃且无任何引用，17 个函数/resolver 在全仓库（含 YAML 配置）中无调用者。这些死代码膨胀了 utils 的感知 API 面，干扰维护者判断模块职责边界，并引入了 `psutil`、`pyarrow` 等未声明的幽灵依赖。

## What Changes

### 整文件删除（2 个文件，~546 行）

- **删除** `src/utils/restart_job.py` — `RestartAndLoadCheckpointCallback`、`BaseJobLauncher`、`LocalJobLauncher`，全部标注 "Deprecated"，无 Python 导入、无 YAML `_target_` 引用
- **删除** `src/utils/restart_job_utils.py` — `JobCheckpointMetadata`、`RestartMetadata` 及 5 个辅助函数，唯一调用者是 `restart_job.py`

### 函数级删除（5 个文件，~270 行）

- `src/utils/custom_hydra_resolvers.py` — 删除 6 个未使用 resolver（`remove_chars_from_string`、`conditional_expression`、`extract_fields_from_list_of_dicts`、`create_map_from_list_of_dicts`、`math_eval`、`remove_item_from_list`）及其注册语句；保留 `now_of_timezone`（8 个 YAML 配置在用）；清理随之多余的 `ast`、`operator`、`DictConfig`、`ListConfig` 导入
- `src/utils/tensor_utils.py` — 删除 `locations_to_index_tuple`、`extract_locations`、`transpose_tensor_from_file`（均无调用者）
- `src/utils/utils.py` — 删除 `get_var_if_not_none`、`get_class_name_str`、`lightning_precision_to_dtype`、`sample_gumbel`、`gumbel_softmax_sample`（均无调用者）；清理随之多余的 `import torch.nn.functional as F`
- `src/utils/file_utils.py` — 删除 `open_pyarrow_file`、`replace_char_after_segment`（均无调用者）；清理随之多余的 `from pyarrow import fs as pyarrow_fs`
- `src/utils/logging_utils.py` — 删除 `convert_dict_to_json_string`（无调用者）；清理随之多余的 `import json`

## Capabilities

### New Capabilities

- `utils-dead-code-removal`: 约定 `src/utils/` 包不得保留无调用者的废弃模块、函数和 resolver，删除后各文件仅保留活跃 API

### Modified Capabilities

（无 — 本次变更不改变任何现有 spec 的行为约定，仅删除无引用代码）

## Impact

- **代码**：7 个文件受影响（2 个删除、5 个精简），不触及任何活跃调用路径
- **依赖**：删除 `restart_job_utils.py` 后，`psutil` 的幽灵依赖消失；删除 `open_pyarrow_file` 后，`pyarrow` 的幽灵依赖消失
- **配置**：无 YAML 配置变更，所有 `_target_` 引用的符号均被保留
- **向后兼容**：所有被删符号在变更前已无调用者，无破坏性影响
