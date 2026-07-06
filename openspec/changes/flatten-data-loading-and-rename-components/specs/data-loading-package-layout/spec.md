## ADDED Requirements

### Requirement: 数据加载包 SHALL 采用扁平化目录布局
`src/data/` 包 SHALL 直接承载 `components/`、`datamodules/`、`utils.py`，且 SHALL NOT 保留 `loading/` 中间层目录。

#### Scenario: 不存在 loading 中间层
- **WHEN** 维护者检查 `src/data/` 目录结构
- **THEN** 目录下 MUST 直接包含 `components/`、`datamodules/`、`utils.py`
- **THEN** 目录下 MUST NOT 包含 `loading/` 子目录

#### Scenario: Python 模块路径不经过 loading
- **WHEN** 仓库代码或配置以 `_target_` 或 import 引用数据加载模块
- **THEN** 模块路径 MUST 形如 `src.data.components.*`、`src.data.datamodules.*`、`src.data.utils`
- **THEN** 模块路径 MUST NOT 形如 `src.data.loading.*`

### Requirement: components 模块文件名 SHALL 准确反映内容职责
`src/data/components/` 下的模块文件名 SHALL 与其内容职责对应，并遵循复数、去下划线的统一命名风格。

#### Scenario: 数据模型模块命名
- **WHEN** 维护者查找数据配置类与批数据容器 dataclass
- **THEN** 这些定义 MUST 位于 `src/data/components/data_models.py`
- **THEN** 该目录 MUST NOT 存在名为 `interfaces.py` 的模块

#### Scenario: 数据集模块命名
- **WHEN** 维护者查找 IterableDataset 定义
- **THEN** 这些定义 MUST 位于 `src/data/components/datasets.py`
- **THEN** 该目录 MUST NOT 存在名为 `dataloading.py` 的模块

#### Scenario: 数据加载器模块命名
- **WHEN** 维护者查找带重试机制的 DataLoader 实现
- **THEN** 该实现 MUST 位于 `src/data/components/dataloaders.py`
- **THEN** 该目录 MUST NOT 存在名为 `custom_dataloader.py` 的模块

#### Scenario: collate 函数模块命名
- **WHEN** 维护者查找 collate 函数
- **THEN** 这些函数 MUST 位于 `src/data/components/collate.py`
- **THEN** 该目录 MUST NOT 存在名为 `collate_functions.py` 的模块

#### Scenario: 标签函数模块命名
- **WHEN** 维护者查找标签或掩码函数
- **THEN** 这些定义 MUST 位于 `src/data/components/label_functions.py`
- **THEN** 该目录 MUST NOT 存在名为 `label_function.py` 的模块

#### Scenario: 预处理函数模块命名
- **WHEN** 维护者查找预处理函数
- **THEN** 这些函数 MUST 位于 `src/data/components/preprocessing.py`
- **THEN** 该目录 MUST NOT 存在名为 `pre_processing.py` 的模块

#### Scenario: 迭代器模块命名保留
- **WHEN** 维护者查找原始文件数据迭代器
- **THEN** 这些定义 MUST 位于 `src/data/components/iterators.py`

### Requirement: 序列数据集类 SHALL 命名为 SequenceDataset
`src/data/components/datasets.py` 中的无界序列 IterableDataset 实现类 SHALL 命名为 `SequenceDataset`。

#### Scenario: 类名简洁且以 Dataset 结尾
- **WHEN** 维护者检查序列数据集实现类
- **THEN** 该类 MUST 命名为 `SequenceDataset`
- **THEN** 代码与配置中 MUST NOT 出现 `UnboundedSequenceIterable` 旧类名
