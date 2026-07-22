## MODIFIED Requirements

### Requirement: data components 中的配置类与批数据类 SHALL 分离定义
`src/data/components/` 中用于描述数据管线配置的 dataclass 与用于承载运行时 batch / label / model input 的 dataclass SHALL 位于不同模块，且不得继续混放在同一个文件中。

#### Scenario: 配置 dataclass 位于专属模块
- **WHEN** 维护者查找 dataset、dataloader 或 tokenizer 的配置 dataclass
- **THEN** 这些定义 MUST 位于 `src/data/components/config_models.py`
- **THEN** `config_models.py` MUST 包含 `DatasetConfig`、`SequenceDataloaderConfig`、`TokenizerConfig`、`ItemDataloaderConfig`
- **AND** `config_models.py` MUST NOT 包含重复的 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`

#### Scenario: 运行时批数据 dataclass 位于专属模块
- **WHEN** 维护者查找 label function 输出、模型输入或 item batch 容器 dataclass
- **THEN** 这些定义 MUST 位于 `src/data/components/data_models.py`
- **THEN** `data_models.py` MUST 包含 `GeneratedLabels`、`TigerLabelData`、`TigerModelInput`、`ItemData`、`ItemTextData`

#### Scenario: 不再保留混合 data_models 入口
- **WHEN** 维护者检查 `src/data/components/` 目录
- **THEN** 目录中 MUST NOT 存在同时混放配置类与批数据类的 `data_models.py`

### Requirement: 角色专属引用路径 SHALL 与模块职责一致
代码与配置对这些 dataclass 的引用 SHALL 指向其对应的角色专属模块路径。

#### Scenario: Hydra targets 指向配置模型模块
- **WHEN** `configs/data/*.yaml` 通过 `_target_` 引用 dataset 或 dataloader 配置类
- **THEN** `_target_` MUST 形如 `src.data.components.config_models.*`
- **THEN** `_target_` MUST NOT 指向 `src.data.components.data_models.*`

#### Scenario: Python import 指向批数据模块
- **WHEN** 代码导入 `GeneratedLabels`、`TigerModelInput`、`TigerLabelData`、`ItemData` 或 `ItemTextData`
- **THEN** import MUST 来自 `src.data.components.data_models`

#### Scenario: Python import 指向配置模型模块
- **WHEN** 代码导入 `DatasetConfig`、`TokenizerConfig` 或其他 dataset / dataloader 配置 dataclass
- **THEN** import MUST 来自 `src.data.components.config_models`
