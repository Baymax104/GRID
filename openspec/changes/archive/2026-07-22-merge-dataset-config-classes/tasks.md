## 1. 合并配置类实现

- [x] 1.1 在 `src/data/components/config_models.py` 中新增统一 `DatasetConfig`，字段保持为 `data_reader`、`preprocessing_functions`、`shuffle_files`
- [x] 1.2 删除 `SemanticIDDatasetConfig` 与 `ItemDatasetConfig` 类定义及重复 docstring
- [x] 1.3 将 `SequenceDataloaderConfig.dataset_config` 与 `ItemDataloaderConfig.dataset_config` 类型注解改为 `DatasetConfig`

## 2. 迁移 Hydra 配置

- [x] 2.1 将 TIGER train/inference data YAML 的 dataset config `_target_` 从 `SemanticIDDatasetConfig` 改为 `DatasetConfig`
- [x] 2.2 将 item/embedding/quantization data YAML 的 dataset config `_target_` 从 `ItemDatasetConfig` 改为 `DatasetConfig`
- [x] 2.3 确认 `configs/data/*.yaml` 中不再引用 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`

## 3. 同步 living specs

- [x] 3.1 更新 `openspec/specs/data-model-role-separation/spec.md` 的配置类清单与 import/target contract
- [x] 3.2 更新 `openspec/specs/data-config-class-convention/spec.md` 的 dataclass convention
- [x] 3.3 更新 `openspec/specs/tiger-sequence-data-contract/spec.md`、`item-dataset-config-runtime-fields-only/spec.md`、`data-reader-factory-contract/spec.md` 中旧类名引用
- [x] 3.4 新增或归档后生成 `unified-dataset-config-contract` living spec delta

## 4. 验证

- [x] 4.1 运行 grep 确认非 archive 的 Python/YAML/living specs 中无 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig` 运行时引用
- [x] 4.2 运行 `uv run python -m compileall -q src/data/components/config_models.py src/data/data_module.py src/data/datasets.py`
- [x] 4.3 运行 `uv run ruff check src/data/components/config_models.py src/data/data_module.py src/data/datasets.py`
- [x] 4.4 对 7 个官方 experiments 执行 Hydra compose + datamodule instantiate smoke，确认 dataset config 使用 `DatasetConfig`
- [x] 4.5 运行 `openspec validate merge-dataset-config-classes --strict`
- [x] 4.6 运行 `openspec validate --specs --no-interactive`
- [x] 4.7 运行 `git diff --check`
