## MODIFIED Requirements

### Requirement: 工具函数 SHALL 在无调用者时删除
`src/utils/` 各模块中在仓库内（Python 导入 + YAML `_target_` + OmegaConf resolver）无任何调用者的函数 SHALL 被删除。

#### Scenario: inference utils 死函数被删除
- **WHEN** 维护者在仓库中搜索 `locations_to_index_tuple`、`extract_locations`、`transpose_tensor_from_file`
- **THEN** `src/inference/utils.py` MUST NOT 包含这些函数定义

#### Scenario: utils.py 死函数被删除
- **WHEN** 维护者在仓库中搜索 `get_var_if_not_none`、`get_class_name_str`、`lightning_precision_to_dtype`、`sample_gumbel`、`gumbel_softmax_sample`
- **THEN** `src/utils/utils.py` MUST NOT 包含这些函数定义

#### Scenario: file_utils 死函数被删除
- **WHEN** 维护者在仓库中搜索 `open_pyarrow_file`、`replace_char_after_segment`、`copy_to_remote`、`file_exists_local_or_remote`、`remove_file_extension`
- **THEN** `src/utils/file_utils.py` MUST NOT 包含这些函数定义

#### Scenario: logging_utils 死函数被删除
- **WHEN** 维护者在仓库中搜索 `convert_dict_to_json_string`
- **THEN** `src/utils/logging_utils.py` MUST NOT 包含该函数定义

### Requirement: 删除操作 SHALL 清理孤立 import
因函数删除而变为无引用的 import 语句 SHALL 被一并清理，确保通过 Ruff F401/I001 检查。

#### Scenario: custom_hydra_resolvers 清理孤立 import
- **WHEN** 维护者检查 `custom_hydra_resolvers.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import ast`、`import operator as op`、`from omegaconf import DictConfig` 或 `from omegaconf import ListConfig`
- **THEN** 文件 MUST 保留 `from datetime import datetime`、`import pytz`、`from omegaconf import OmegaConf`

#### Scenario: utils.py 清理孤立 import
- **WHEN** 维护者检查 `utils.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import torch.nn.functional as F`

#### Scenario: file_utils.py 清理孤立 import
- **WHEN** 维护者检查 `file_utils.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `from pyarrow import fs as pyarrow_fs`
- **THEN** 文件 MUST NOT 包含 `from shutil import SameFileError`
- **THEN** 文件 MUST NOT 包含 `from lightning.fabric.utilities.types import _PATH`

#### Scenario: logging_utils.py 清理孤立 import
- **WHEN** 维护者检查 `logging_utils.py` 的 import 语句
- **THEN** 文件 MUST NOT 包含 `import json`
