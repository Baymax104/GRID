## Why

`src/data/components/data_models.py` 当前同时承载两类概念：一类是供 Hydra / preprocessing / dataloader 装配使用的配置 dataclass，另一类是供 collate / label function / model forward 使用的批数据 dataclass。两类对象生命周期、消费者、依赖都不同，却混在同一个文件里，导致文件职责过宽，导航时很难快速判断某个符号属于“配置”还是“运行时 batch 数据”。

前一轮重构已将该文件从 `interfaces.py` 更名为 `data_models.py`，但并未解决“配置类与数据类同居一处”的职责混杂问题。现在引用面已基本摸清，适合继续按角色拆开，降低后续维护与继续演进的认知负担。

## What Changes

- **BREAKING（内部）** 将 `src/data/components/data_models.py` 拆分为两个职责单一的模块：
  - `src/data/components/config_models.py`：仅存放配置 dataclass
  - `src/data/components/data_models.py`：仅存放运行时批数据 dataclass
- 配置类迁移到 `config_models.py`：`SequenceDatasetConfig`、`SequenceDataloaderConfig`、`SemanticIDDatasetConfig`、`TokenizerConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig`
- 数据类迁移到 `data_models.py`：`LabelFunctionOutput`、`SequentialModuleLabelData`、`SequentialModelInputData`、`ItemData`、`ItemTextData`
- 同步更新全部 Python import 与 `configs/data/*.yaml` 的 Hydra `_target_` 路径，使引用落到对应的新模块
- 删除旧的混合模块 `src/data/components/data_models.py`，避免继续作为“兜底聚合文件”存在

## Capabilities

### New Capabilities
- `data-model-role-separation`: 规定 data components 中的配置 dataclass 与批数据 dataclass 必须按职责分离到不同模块

## Impact

- 受影响源码：
  - `src/data/components/data_models.py`（删除并拆分）
  - `src/data/components/preprocessing.py`、`src/utils/utils.py`（配置类 import）
  - `src/data/components/collate.py`、`src/data/components/label_functions.py`、`src/embedding/semantic_embedding_inference_module.py`、`src/quantization/residual_quantization.py`、`src/recommendation/base_recommender.py`、`src/recommendation/tiger_generation_model.py`、`src/common/modules/transformer_base_module.py`（数据类 import）
- 受影响配置：`configs/data/*.yaml` 中指向 `SequenceDataloaderConfig`、`SemanticIDDatasetConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig` 的 `_target_`
- 不改变任何 dataclass 字段与运行时行为；仅调整模块边界与引用路径
