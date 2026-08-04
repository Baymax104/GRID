# data-config-class-convention Specification

## Purpose
TBD - created by archiving change remove-empty-config-base-classes. Update Purpose after archive.
## Requirements
### Requirement: data config 类 SHALL NOT 定义仅为类型标记的空基类
data config dataclass SHALL NOT 继承仅为提供类型标记或单一辅助方法（如 `.get()`）而存在的空基类。config 类 SHALL 直接以 `@dataclass` 定义，所需的可选字段访问 SHALL 使用 Python 内置 `getattr` 实现。

#### Scenario: 不存在空 config 基类
- **WHEN** 维护者检查 `src/common/configs/data.py`
- **THEN** 该文件 MUST NOT 定义 `BaseDatasetConfig` 或 `BaseDataloaderConfig`
- **THEN** `DatasetConfig`、`SequenceDataloaderConfig`、`ItemDataloaderConfig` MUST 直接以 `@dataclass` 定义，不继承被删基类

#### Scenario: 访问可选 config 字段用 getattr
- **WHEN** 代码访问 config 对象的可选字段并需要默认值
- **THEN** 代码 MUST 使用 `getattr(obj, attr, default)` 而非依赖自定义 `.get()` 方法

### Requirement: dataset/dataloader config 的类型注解 SHALL 使用 DictConfig
data 模块中消费 dataset config 与 dataloader config 对象的函数/方法参数类型注解 SHALL 使用 `omegaconf.DictConfig`，而非已删除的自定义基类类型。其他独立 config dataclass 的注解不受此约束。

#### Scenario: dataset_config / dataloader_config 注解统一
- **WHEN** 维护者检查 `src/data/` 与 `src/data/components/` 中消费 dataset config 或 dataloader config 的参数注解
- **THEN** 该注解 MUST 为 `DictConfig`
- **THEN** 代码 MUST NOT 引用 `BaseDatasetConfig` 或 `BaseDataloaderConfig` 作为类型注解

