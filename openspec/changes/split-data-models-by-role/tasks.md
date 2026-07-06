## 1. 拆分 `data_models.py`

- [x] 1.1 新建 `src/data/components/config_models.py`，迁移 6 个配置 dataclass：`SequenceDatasetConfig`、`SequenceDataloaderConfig`、`SemanticIDDatasetConfig`、`TokenizerConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig`
- [x] 1.2 新建 `src/data/components/data_models.py`，迁移 5 个数据 dataclass：`LabelFunctionOutput`、`SequentialModuleLabelData`、`SequentialModelInputData`、`ItemData`、`ItemTextData`
- [x] 1.3 保持全部类名、字段顺序、默认值、文档字符串与继承关系不变
- [x] 1.4 删除旧文件 `src/data/components/data_models.py`

## 2. 更新 Python import

- [x] 2.1 `src/utils/utils.py`、`src/data/components/preprocessing.py`：`TokenizerConfig` import 改到 `config_models.py`
- [x] 2.2 `src/data/components/collate.py`、`src/data/components/label_functions.py`：数据类 import 改到 `data_models.py`
- [x] 2.3 `src/embedding/semantic_embedding_inference_module.py`、`src/quantization/residual_quantization.py`：`ItemData` import 改到 `data_models.py`
- [x] 2.4 `src/recommendation/base_recommender.py`、`src/recommendation/tiger_generation_model.py`、`src/common/modules/transformer_base_module.py`：`SequentialModelInputData` / `SequentialModuleLabelData` import 改到 `data_models.py`

## 3. 更新 Hydra `_target_`

- [x] 3.1 更新 `configs/data/*.yaml` 中 `SequenceDataloaderConfig` 的 `_target_` 到 `src.data.components.config_models.SequenceDataloaderConfig`
- [x] 3.2 更新 `configs/data/*.yaml` 中 `SemanticIDDatasetConfig` 的 `_target_` 到 `src.data.components.config_models.SemanticIDDatasetConfig`
- [x] 3.3 更新 `configs/data/*.yaml` 中 `ItemDatasetConfig` 与 `ItemDataloaderConfig` 的 `_target_` 到 `src.data.components.config_models.*`

## 4. 验证收尾

- [x] 4.1 `rg` 复查仓库（排除 `openspec/`）确认 `src.data.components.data_models` 零代码/配置残留
- [x] 4.2 import smoke check：导入全部直接受影响模块，确认无循环导入或路径错误
- [x] 4.3 Hydra compose / 最小实例化检查：确认 7 个 `configs/experiment/*.yaml` 相关 data 配置仍能解析到新的 `_target_`
