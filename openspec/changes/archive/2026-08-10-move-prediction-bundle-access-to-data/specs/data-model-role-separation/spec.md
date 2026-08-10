## MODIFIED Requirements

### Requirement: data components 中的配置类与批数据类 SHALL 分离定义
用于描述数据管线配置的 dataclass 与用于承载运行时 batch / label / model input / keyed prediction output 的 dataclass SHALL 位于不同模块，且不得继续混放在同一个文件中。数据配置 dataclass SHALL 位于 common config package，运行时数据 dataclass SHALL 留在 data package。

#### Scenario: 配置 dataclass 位于 common configs 专属模块
- **WHEN** 维护者查找 dataset 或 dataloader 的配置 dataclass
- **THEN** 这些定义 MUST 位于 `src/common/configs/data.py`
- **THEN** `data.py` MUST 包含 `DatasetConfig`、`SequenceDataloaderConfig`、`ItemDataloaderConfig`
- **AND** `src/data/components/config_models.py` MUST NOT 定义这些运行时配置类
- **AND** `src/common/configs/data.py` MUST NOT 包含重复的 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`

#### Scenario: 运行时数据 dataclass 位于专属模块
- **WHEN** 维护者查找 TIGER 模型输入、TIGER label data、item batch 容器或 keyed prediction output 容器 dataclass
- **THEN** 这些定义 MUST 位于 `src/data/components/data_models.py`
- **THEN** `data_models.py` MUST 包含 `TigerLabelData`、`TigerModelInput`、`ItemBatch`、`ModelOutput`
- **THEN** `data_models.py` MUST NOT 包含未使用的 `ItemTextBatch`
- **AND** `data_models.py` MUST NOT 包含旧的通用 sequential batch 或 label output dataclass

#### Scenario: 不再保留混合 data_models 入口
- **WHEN** 维护者检查 `src/data/components/` 目录
- **THEN** 目录中 MUST NOT 存在同时混放配置类与批数据类的 `data_models.py`

### Requirement: 角色专属引用路径 SHALL 与模块职责一致
代码与配置对这些 dataclass 的引用 SHALL 指向其对应的角色专属模块路径。

#### Scenario: Hydra targets 指向 common 配置模型模块
- **WHEN** `configs/data/*.yaml` 通过 `_target_` 引用 dataset 或 dataloader 配置类
- **THEN** `_target_` MUST 形如 `src.common.configs.data.*`
- **THEN** `_target_` MUST NOT 指向 `src.data.components.data_models.*`
- **THEN** `_target_` MUST NOT 指向 `src.data.components.config_models.*`

#### Scenario: Python import 指向运行时数据模块
- **WHEN** 代码导入 `TigerModelInput`、`TigerLabelData`、`ItemBatch` 或 `ModelOutput`
- **THEN** import MUST 来自 `src.data.components.data_models`
- **THEN** import MUST NOT 来自 `src.inference.model_output`

#### Scenario: Python import 指向 common 配置模型模块
- **WHEN** 代码导入 `DatasetConfig` 或其他 dataset / dataloader 配置 dataclass
- **THEN** import MUST 来自 `src.common.configs.data`
