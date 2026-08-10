# utils-dead-code-removal Specification

## Purpose
TBD - created by archiving change delete-utils-dead-code. Update Purpose after archive.
## Requirements
### Requirement: 废弃模块 SHALL 在无引用时删除
`src/utils/` 中标注为 Deprecated 且在仓库内（Python 代码 + YAML 配置）无任何引用的模块 SHALL 被整体删除，不得以 stub 形式保留。

#### Scenario: restart_job 模块无引用时被删除
- **WHEN** 维护者在仓库中搜索 `restart_job` 或 `RestartAndLoadCheckpointCallback` 或 `BaseJobLauncher` 或 `LocalJobLauncher`
- **THEN** 在 `src/utils/` 下 MUST NOT 存在 `restart_job.py` 文件
- **THEN** 在 `src/utils/` 下 MUST NOT 存在 `restart_job_utils.py` 文件

### Requirement: 自定义 resolver SHALL 仅保留有配置引用的项
`custom_hydra_resolvers.py` 中注册的 OmegaConf resolver SHALL 仅保留在 `configs/**/*.yaml` 中有实际 `${resolver_name:...}` 引用的项。无配置引用的 resolver 函数及其注册语句 SHALL 被删除。

#### Scenario: now_tz resolver 被保留
- **WHEN** 维护者检查 `custom_hydra_resolvers.py`
- **THEN** 文件 MUST 包含 `now_of_timezone` 函数定义
- **THEN** 文件 MUST 包含 `OmegaConf.register_new_resolver("now_tz", now_of_timezone)` 注册语句

#### Scenario: 无引用 resolver 被删除
- **WHEN** 维护者在 `configs/**/*.yaml` 中搜索 `${remove_chars_from_string`、`${conditional_expression`、`${extract_fields_from_list_of_dicts`、`${create_map_from_list_of_dicts`、`${math_eval`、`${remove_item_from_list`
- **THEN** 搜索结果 MUST 为空
- **THEN** `custom_hydra_resolvers.py` MUST NOT 包含这些 resolver 的函数定义或注册语句

### Requirement: 工具函数 SHALL 在无调用者时删除
`src/utils/` 各模块中在仓库内（Python 导入 + YAML `_target_` + OmegaConf resolver）无任何调用者的函数 SHALL 被删除。

#### Scenario: 旧 inference utils 死函数被删除
- **WHEN** 维护者在仓库中搜索 `locations_to_index_tuple`、`extract_locations`、`transpose_tensor_from_file`
- **THEN** 仓库中 MUST NOT 包含这些函数定义

#### Scenario: utils.py 死函数被删除
- **WHEN** 维护者在仓库中搜索 `get_var_if_not_none`、`get_class_name_str`、`lightning_precision_to_dtype`、`sample_gumbel`、`gumbel_softmax_sample`
- **THEN** `src/utils/utils.py` MUST NOT 包含这些函数定义

#### Scenario: file.py 死函数被删除
- **WHEN** 维护者在仓库中搜索 `open_pyarrow_file`、`replace_char_after_segment`
- **THEN** `src/utils/file.py` MUST NOT 包含这些函数定义

#### Scenario: logging.py 死函数被删除
- **WHEN** 维护者在仓库中搜索 `convert_dict_to_json_string`
- **THEN** `src/utils/logging.py` MUST NOT 包含该函数定义

### Requirement: 删除操作 SHALL 清理孤立 import
因函数删除而变为无引用的 import 语句 SHALL 被一并清理，确保通过 Ruff F401/I001 检查。

#### Scenario: custom_hydra_resolvers 清理孤立 import
- **WHEN** 维护者检查 `custom_hydra_resolvers.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import ast`、`import operator as op`、`from omegaconf import DictConfig` 或 `from omegaconf import ListConfig`
- **THEN** 文件 MUST 保留 `from datetime import datetime`、`import pytz`、`from omegaconf import OmegaConf`

#### Scenario: utils.py 清理孤立 import
- **WHEN** 维护者检查 `utils.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import torch.nn.functional as F`

#### Scenario: file.py 清理孤立 import
- **WHEN** 维护者检查 `file.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `from pyarrow import fs as pyarrow_fs`

#### Scenario: logging.py 清理孤立 import
- **WHEN** 维护者检查 `logging.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import json`
