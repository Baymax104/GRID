## Why

`BaseDatasetConfig` 与 `BaseDataloaderConfig` 是两个仅含 `__init__: pass` 与一个 `.get()` 方法的空基类，文档自述仅 "provide base typing"。全仓无任何 `isinstance` 检查依赖它们，`.get()` 仅 4 处调用且可被 `getattr` 平替。子类 dataclass 去继承后由 `@dataclass` 生成 `__init__`，构造行为不变。这两个基类是纯噪音，徒增继承层级而无实际约束力。

同时，config 类型注解在代码中不一致：`base.py` 标注 `DictConfig`，而 `datasets.py` / `preprocessing.py` 标注 `BaseDatasetConfig`。实际运行时 config 是 Hydra 实例化的 dataclass 对象，消费方按鸭子类型（属性访问 + `.get()`）访问，与 `DictConfig` 完全兼容。统一为 `DictConfig` 注解可消除不一致，并反映"消费方按鸭子类型访问"的真实设计意图。

## What Changes

- **BREAKING（内部）** 删除 `BaseDatasetConfig`、`BaseDataloaderConfig` 两个类定义。
- 4 个直接继承被删基类的 dataclass 子类去掉继承：`SequenceDatasetConfig`、`SequenceDataloaderConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig`。`SemanticIDDatasetConfig(SequenceDatasetConfig)` 继承的是中间类 `SequenceDatasetConfig`，继承链不断，不变。
- `preprocessing.py` 与 `datasets.py` 中引用被删基类的 config 类型注解改为 `DictConfig`。`preprocessing.py` 的 `TokenizerConfig` 注解保留不动（独立 dataclass，不在本次范围）。
- 4 处 `.get(attr, default)` 改为 `getattr(obj, attr, default)`（`base.py:66`、`sequence.py:35`、`preprocessing.py:57,60`），因去继承后 dataclass 实例不再继承 `.get()` 方法。
- 调整 import：`preprocessing.py` 保留 `TokenizerConfig` import、新增 `DictConfig` import；`datasets.py` 删除 `BaseDatasetConfig` import、新增 `DictConfig` import。

## Capabilities

### New Capabilities
- `data-config-class-convention`: 规定 data 模块 config 类的设计约定——不定义仅为类型标记的空基类；消费 config 对象的类型注解统一用 `DictConfig`（反映鸭子类型访问的真实意图）。

## Impact

- 受影响代码：`src/data/components/data_models.py`（删 2 类 + 4 子类去继承）、`src/data/components/preprocessing.py`（9 处注解换 DictConfig + 2 处 `.get` 改 getattr + import 调整）、`src/data/components/datasets.py`（2 处注解 + 1 处文档 + import 调整）、`src/data/datamodules/base.py`（1 处 `.get` 改 getattr）、`src/data/datamodules/sequence.py`（1 处 `.get` 改 getattr）
- 不影响 yaml 配置、不影响运行时行为、不引入新依赖
- `base.py` 的类型注解已是 `DictConfig`，不动
