## Context

`BaseDatasetConfig` 与 `BaseDataloaderConfig` 是两个空基类（仅 `__init__: pass` + 一个 `.get()` 方法，文档自述 "provide base typing"）。全仓无 `isinstance` 检查依赖它们。`.get()` 仅 4 处调用，等价于 `getattr`。

config 类型注解在代码中不一致：`base.py` 标注 `DictConfig`，而 `datasets.py` / `preprocessing.py` 标注 `BaseDatasetConfig`。实际运行时 config 是 Hydra 实例化的 dataclass 对象，消费方按鸭子类型（属性访问 + `.get()`）访问，与 `DictConfig` 兼容。

子类继承结构：`SequenceDatasetConfig`、`SequenceDataloaderConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig` 直接继承被删基类；`SemanticIDDatasetConfig(SequenceDatasetConfig)` 继承中间类 `SequenceDatasetConfig`；`TokenizerConfig` 独立、不继承任何基类。

## Goals / Non-Goals

**Goals:**
- 删除 `BaseDatasetConfig`、`BaseDataloaderConfig` 两个空基类
- 4 个直接继承被删基类的子类去掉继承
- `preprocessing.py` 与 `datasets.py` 中引用被删基类的 config 注解统一为 `DictConfig`
- 4 处 `.get(attr, default)` 改为 `getattr(obj, attr, default)`

**Non-Goals:**
- 不改 `TokenizerConfig`（独立 dataclass，注解保留）
- 不改 `SemanticIDDatasetConfig(SequenceDatasetConfig)` 的继承（继承中间类，不在去继承范围）
- 不改 `base.py` 的类型注解（已为 `DictConfig`）
- 不改 yaml 配置
- 不引入 `typing.Protocol` 或 `ABC` 替代基类

## Decisions

### D1: 直接删除基类，不引入 Protocol/ABC 替代
- **选择**：删除两个空基类，子类去继承，不引入任何替代类型。
- **理由**：基类仅提供 `.get()`（可 `getattr` 平替）和类型标记（无 `isinstance` 依赖）。引入 `Protocol` 替代会增加抽象而无功能收益——消费方本就按鸭子类型访问。
- **备选**：用 `typing.Protocol` 定义 config 协议——否决，过度设计。

### D2: 注解统一 `DictConfig` 而非具体子类类型
- **选择**：消费 dataset/dataloader config 的注解统一用 `DictConfig`。
- **理由**：`base.py` 已用 `DictConfig`；实际运行时 config 是 dataclass 实例但消费方按属性访问（鸭子类型）；`DictConfig` 注解与访问方式一致，且与 `base.py` 统一。
- **备选**：用具体子类类型（如 `SequenceDatasetConfig`）——更准确但与 `base.py` 不一致，且消费方不依赖具体类型；用 `Any`——丢失语义。均否决。

### D3: `.get()` → `getattr` 而非给 dataclass 加 `.get()`
- **选择**：4 处 `.get(attr, default)` 改为 `getattr(obj, attr, default)`。
- **理由**：去继承后 dataclass 无 `.get()`。`getattr` 是 Python 内置，等价且无需新增方法。
- **备选**：给每个 dataclass 加 `.get()` 方法——否决，散布样板代码，违背 D1。

### D4: `SemanticIDDatasetConfig` 继承不动
- **选择**：`SemanticIDDatasetConfig(SequenceDatasetConfig)` 保持不变。
- **理由**：去继承指去掉对**被删基类**的直接继承。`SemanticIDDatasetConfig` 继承的是中间类 `SequenceDatasetConfig`，后者去掉对 `BaseDatasetConfig` 的继承后，`SemanticIDDatasetConfig` 仍继承 `SequenceDatasetConfig`，链不断、字段全部保留。

## Risks / Trade-offs

- **[风险] 去继承后遗漏某处 `.get()` 调用** → 缓解：`rg` 全仓复查 data 模块的 `.get(`，确认仅 4 处且全改 `getattr`；smoke check 时 import + Hydra compose 验证。
- **[权衡] `DictConfig` 注解与实际 dataclass 实例类型不符** → 可接受，与 `base.py` 现状一致，反映鸭子类型访问意图；运行时不强制类型，且 `DictConfig` 与 dataclass 都支持属性访问。
