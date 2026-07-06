## 1. 展平目录结构（消除 loading 中间层）

- [x] 1.1 用 `git mv` 将 `src/data/loading/components/` 移到 `src/data/components/`
- [x] 1.2 用 `git mv` 将 `src/data/loading/datamodules/` 移到 `src/data/datamodules/`
- [x] 1.3 用 `git mv` 将 `src/data/loading/utils.py` 移到 `src/data/utils.py`
- [x] 1.4 删除空的 `src/data/loading/__init__.py` 与 `src/data/loading/` 目录
- [x] 1.5 新增空 `src/data/__init__.py`，与同级包结构一致

## 2. 更新展平后的内部 import（路径前缀 `loading.` 去除）

- [x] 2.1 更新 `src/data/datamodules/__init__.py` 的 3 处 import 路径
- [x] 2.2 更新 `src/data/datamodules/base.py`（`components.custom_dataloader`、`utils` 引用）
- [x] 2.3 更新 `src/data/datamodules/sequence.py` 与 `item.py`（`base` 引用）
- [x] 2.4 更新 `src/data/components/` 下相互引用（`interfaces`→`iterators`、`collate_functions`→`interfaces`+`utils`、`dataloading`→`interfaces`、`pre_processing`→`interfaces`、`label_function`→`interfaces`）

## 3. 更新外部 Python 引用与配置（展平阶段）

- [x] 3.1 更新 6 个外部模块的 import：`src.data.loading.components.interfaces` → `src.data.components.interfaces`（`src/common/modules/transformer_base_module.py`、`src/utils/utils.py`、`src/embedding/semantic_embedding_inference_module.py`、`src/quantization/residual_quantization.py`、`src/recommendation/base_recommender.py`、`src/recommendation/tiger_generation_model.py`）
- [x] 3.2 更新 7 个 `configs/data/*.yaml` 的 `_target_`：`src.data.loading.components.*` → `src.data.components.*`、`src.data.loading.datamodules.*` → `src.data.datamodules.*`
- [x] 3.3 展平阶段 smoke check：`rg` 确认 `src.data.loading` 零残留，import 与 Hydra compose 通过

## 4. 重命名 components 模块文件

- [x] 4.1 `git mv` `interfaces.py` → `data_models.py`
- [x] 4.2 `git mv` `dataloading.py` → `datasets.py`
- [x] 4.3 `git mv` `custom_dataloader.py` → `dataloaders.py`
- [x] 4.4 `git mv` `collate_functions.py` → `collate.py`
- [x] 4.5 `git mv` `label_function.py` → `label_functions.py`
- [x] 4.6 `git mv` `pre_processing.py` → `preprocessing.py`
- [x] 4.7 `iterators.py` 保留不动

## 5. 重命名类与更新重命名后的引用

- [x] 5.1 在 `datasets.py` 中将类 `UnboundedSequenceIterable` 重命名为 `SequenceDataset`
- [x] 5.2 更新 `src/data/components/` 内部 import 到新文件名（`datasets`→`data_models`、`collate`→`data_models`+`utils`、`label_functions`→`data_models`、`preprocessing`→`data_models`、`dataloaders` 无内部依赖）
- [x] 5.3 更新 6 个外部模块的 import 到新文件名（`interfaces` → `data_models`）
- [x] 5.4 更新 7 个 `configs/data/*.yaml` 的 `_target_` 到新文件名与新类名（`dataloading.UnboundedSequenceIterable` → `datasets.SequenceDataset`，其余 components 模块名同步更新）
- [x] 5.5 重命名阶段 smoke check：`rg` 确认旧文件名与旧类名零残留

## 6. 更新文档与收尾

- [x] 6.1 更新 `AGENTS.md` 中 `src/data/loading/` 的描述为扁平化后的路径
- [x] 6.2 更新 `src/data/README.md` 中对模块路径的描述（修正过时的 `datamodules.base.BaseFileDataModule` 为实际类名 `BaseDataModule` 与新路径）
- [x] 6.3 全量 smoke check：import 全部 data 子模块、Hydra compose 7 个 experiment 配置、`rg` 复查 `data\.loading|UnboundedSequenceIterable|interfaces|dataloading\.py|custom_dataloader|collate_functions|label_function\.py|pre_processing` 零残留
