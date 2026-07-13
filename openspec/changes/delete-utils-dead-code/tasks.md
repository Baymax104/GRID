## 1. 删除废弃整文件

- [ ] 1.1 删除 `src/utils/restart_job.py`
- [ ] 1.2 删除 `src/utils/restart_job_utils.py`

## 2. 精简 custom_hydra_resolvers.py

- [ ] 2.1 删除 6 个未使用 resolver 函数定义（`remove_chars_from_string`、`conditional_expression`、`extract_fields_from_list_of_dicts`、`create_map_from_list_of_dicts`、`math_eval`、`remove_item_from_list`）
- [ ] 2.2 删除对应的 6 条 `OmegaConf.register_new_resolver` 注册语句
- [ ] 2.3 清理孤立 import：`import ast`、`import operator as op`、`from omegaconf import DictConfig, ListConfig`（保留 `OmegaConf`）
- [ ] 2.4 删除模块级 docstring 中与已删 resolver 相关的说明

## 3. 精简 tensor_utils.py

- [ ] 3.1 删除 `locations_to_index_tuple` 函数
- [ ] 3.2 删除 `extract_locations` 函数
- [ ] 3.3 删除 `transpose_tensor_from_file` 函数

## 4. 精简 utils.py

- [ ] 4.1 删除 `get_var_if_not_none` 函数
- [ ] 4.2 删除 `get_class_name_str` 函数
- [ ] 4.3 删除 `lightning_precision_to_dtype` 函数
- [ ] 4.4 删除 `sample_gumbel` 函数
- [ ] 4.5 删除 `gumbel_softmax_sample` 函数
- [ ] 4.6 清理孤立 import：`import torch.nn.functional as F`

## 5. 精简 file_utils.py

- [ ] 5.1 删除 `open_pyarrow_file` 函数
- [ ] 5.2 删除 `replace_char_after_segment` 函数
- [ ] 5.3 清理孤立 import：`from pyarrow import fs as pyarrow_fs`（保留 `from shutil import SameFileError`）

## 6. 精简 logging_utils.py

- [ ] 6.1 删除 `convert_dict_to_json_string` 函数
- [ ] 6.2 清理孤立 import：`import json`

## 7. 验证

- [ ] 7.1 运行 `uv run ruff check src/utils/` 确认无 F401（未使用 import）和 F811（未使用函数）报错
- [ ] 7.2 全仓库搜索被删符号名，确认零残留引用
- [ ] 7.3 确认 `now_tz` resolver 仍存在于 `custom_hydra_resolvers.py` 且 8 个 YAML 配置引用不受影响
