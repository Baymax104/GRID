## Why

当前数据加载源码集中在 `src/data/loading/` 下，但 `loading` 这一层中间目录与 `src/data/` 的语义重复，徒增一层无信息量的嵌套。其下 `components/` 各模块命名也参差不齐：`interfaces.py` 名不副实（内容是配置类与数据容器而非抽象接口）、`dataloading.py` 与上层目录语义重复（实为 dataset 定义）、`custom_dataloader.py` 语义模糊、`label_function.py` 单数与其他复数文件名不一致、`pre_processing.py` 下划线分割不符合惯例。这些都会增加代码导航与维护的认知负担，需要消除 `loading` 中间层并将 components 模块重命名为准确反映内容的名字。

## What Changes

- **BREAKING** 消除 `src/data/loading/` 中间层：将其下的 `components/`、`datamodules/`、`utils.py` 上移到 `src/data/` 下，Python 模块路径从 `src.data.loading.*` 收敛为 `src.data.*`，并删除空的 `src/data/loading/__init__.py`。
- **BREAKING** 重命名 components 各模块以准确反映内容：
  - `interfaces.py` → `data_models.py`（内容为数据配置类与批数据容器 dataclass，非接口）
  - `dataloading.py` → `datasets.py`，其中类 `UnboundedSequenceIterable` → `SequenceDataset`
  - `custom_dataloader.py` → `dataloaders.py`
  - `collate_functions.py` → `collate.py`
  - `label_function.py` → `label_functions.py`
  - `pre_processing.py` → `preprocessing.py`
  - `iterators.py` 保留原名
- 同步更新全部引用点：`configs/data/*.yaml` 的 Hydra `_target_`、外部 Python 模块的 import、`src/data/` 内部相互 import，以及 `AGENTS.md`、`src/data/README.md` 文档。
- 为 `src/data/` 补一个空 `__init__.py`，与 `src/utils/`、`src/common/` 等同级包保持结构一致。

## Capabilities

### New Capabilities
- `data-loading-package-layout`: 规定 `src/data/` 的扁平化目录布局，以及 components 模块文件名与核心类名的命名规范。

### Modified Capabilities
- `datamodule-structure-alignment`: datamodule 的 Hydra `_target_` 模块路径从 `src.data.loading.datamodules.*` 更新为 `src.data.datamodules.*`，以反映扁平化后的布局。

## Impact

- 受影响代码：`src/data/loading/` 下全部文件（移动 + 重命名 + 内部 import 改写），以及外部引用 `src.data.loading.components.interfaces` 的 6 个模块——`src/common/modules/transformer_base_module.py`、`src/utils/utils.py`、`src/embedding/semantic_embedding_inference_module.py`、`src/quantization/residual_quantization.py`、`src/recommendation/base_recommender.py`、`src/recommendation/tiger_generation_model.py`
- 受影响配置：全部 7 个 `configs/data/*.yaml` 的 `_target_` 路径
- 受影响文档：`AGENTS.md`、`src/data/README.md`
- 不引入新依赖，不改变运行时行为与外部实验参数语义，仅改变代码组织与模块路径
