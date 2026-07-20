## 1. 删除空基类与子类去继承（`data_models.py`）

- [x] 1.1 删除 `BaseDatasetConfig` 类定义
- [x] 1.2 删除 `BaseDataloaderConfig` 类定义
- [x] 1.3 `SequenceDatasetConfig(BaseDatasetConfig)` → `SequenceDatasetConfig`（去继承）
- [x] 1.4 `SequenceDataloaderConfig(BaseDataloaderConfig)` → `SequenceDataloaderConfig`（去继承）
- [x] 1.5 `ItemDatasetConfig(BaseDatasetConfig)` → `ItemDatasetConfig`（去继承）
- [x] 1.6 `ItemDataloaderConfig(BaseDataloaderConfig)` → `ItemDataloaderConfig`（去继承）
- [x] 1.7 确认 `SemanticIDDatasetConfig(SequenceDatasetConfig)` 继承不变（继承中间类，不在去继承范围）

## 2. 类型注解统一为 DictConfig

- [x] 2.1 `preprocessing.py`：8 处 `BaseDatasetConfig` 参数注解 + 1 处 `SemanticIDDatasetConfig` 参数注解 → `DictConfig`；`TokenizerConfig` 注解保留不动
- [x] 2.2 `preprocessing.py` import 调整：从 `from src.data.components.data_models import BaseDatasetConfig, SemanticIDDatasetConfig, TokenizerConfig` 改为 `from src.data.components.data_models import TokenizerConfig`，并新增 `from omegaconf import DictConfig`
- [x] 2.3 `datasets.py`：2 处 `dataset_config: BaseDatasetConfig` 参数注解 + 1 处文档注释 → `DictConfig`
- [x] 2.4 `datasets.py` import 调整：删除 `from src.data.components.data_models import BaseDatasetConfig`，新增 `from omegaconf import DictConfig`

## 3. `.get()` → `getattr`

- [x] 3.1 `base.py:66` `config.get("should_shuffle_rows", False)` → `getattr(config, "should_shuffle_rows", False)`
- [x] 3.2 `sequence.py:35` `curr_config.get("oov_token", None)` → `getattr(curr_config, "oov_token", None)`
- [x] 3.3 `preprocessing.py:57` `dataset_config.get("keep_user_id", False)` → `getattr(dataset_config, "keep_user_id", False)`
- [x] 3.4 `preprocessing.py:60` `dataset_config.get("keep_item_id", False)` → `getattr(dataset_config, "keep_item_id", False)`

## 4. 验证收尾

- [x] 4.1 `rg` 复查 `BaseDatasetConfig|BaseDataloaderConfig` 全仓零残留（`*.py`/`*.yaml`/`*.md`，排除 `openspec/`）
- [x] 4.2 `rg` 复查 data 模块 config 上的 `.get(` 调用已全部改为 `getattr`
- [x] 4.3 smoke check：`py_compile` 改动文件 + `import` 全部 data 子模块 + Hydra compose 7 个 experiment 验证 `_target_` 可解析
