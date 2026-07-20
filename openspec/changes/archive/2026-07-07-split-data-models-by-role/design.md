## Context

`src/data/components/data_models.py` 目前包含 11 个 dataclass，其中 6 个是“配置对象”，5 个是“运行时数据容器”。两组类型的消费者完全不同：

- 配置对象主要被 Hydra `_target_`、`preprocessing.py`、`utils.py` 消费
- 数据容器主要被 `collate.py`、`label_functions.py`、embedding / quantization / recommendation 模型消费

内部依赖也呈现两个稳定簇：

- `SemanticIDDatasetConfig` 继承 `SequenceDatasetConfig`
- `ItemTextData` 继承 `ItemData`

这说明按“配置 / 数据”切分有天然边界，且切分后两个文件内部仍能保持紧密内聚。

## Goals / Non-Goals

**Goals:**
- 将配置 dataclass 与运行时数据 dataclass 拆到两个职责单一的文件
- 让 Hydra `_target_` 与 Python import 指向语义更准确的模块名
- 保持所有类名、字段、继承关系与运行时行为不变

**Non-Goals:**
- 不修改任一 dataclass 的字段、默认值、文档语义
- 不新增第三个“兼容转发”模块保留 `data_models.py`
- 不重命名类名
- 不调整 `iterators.py`、`datasets.py`、`dataloaders.py` 等相邻模块职责

## Decisions

### D1: 采用 `config_models.py` + `data_models.py` 两文件切分
- **选择**：配置类进入 `config_models.py`，运行时批数据类进入 `data_models.py`。
- **理由**：这是当前文件里最稳定、最清晰的职责边界。`config_models` 明确表达“可被实例化的配置对象”，`data_models` 延续仓库既有术语，同时承载“批级运行时输入/输出封装”。
- **备选**：`configs.py` / `data.py`——过于宽泛，且 `data.py` 与上层 `src/data/` 易混淆；使用一个全新的“batch-*”命名——能表达批数据，但会额外引入新术语；`containers.py`——能表达容器，但弱化了“batch/model IO”语义。均否决。

### D2: 不保留 `data_models.py` 兼容壳文件
- **选择**：迁移全部引用后直接删除 `src/data/components/data_models.py`。
- **理由**：本仓库引用面已可控枚举；保留壳文件会让职责重新模糊，也会延长旧路径寿命。并且 YAML `_target_` 反正必须更新，无法靠 re-export 兼容。
- **备选**：保留 `data_models.py` 做 re-export——否决，会留下长期双入口。

### D3: 继承关系随角色整体迁移，不跨文件拆散
- **选择**：`SemanticIDDatasetConfig` 与 `SequenceDatasetConfig` 同留在 `config_models.py`；`ItemTextData` 与 `ItemData` 同留在 `data_models.py`。
- **理由**：避免为简单继承引入跨文件导入，保持每个文件内部的局部完整性。

### D4: 所有引用同步切到角色专属入口
- **选择**：
  - 配置消费者统一从 `src.data.components.config_models` 导入
  - 数据消费者统一从 `src.data.components.data_models` 导入
  - Hydra `_target_` 统一改到 `config_models`
- **理由**：只有引用全面切换，拆分后的边界才真正生效。

## Risks / Trade-offs

- **[风险] YAML `_target_` 漏改导致 Hydra instantiate 失败** → 缓解：逐类枚举 `SequenceDataloaderConfig`、`SemanticIDDatasetConfig`、`ItemDatasetConfig`、`ItemDataloaderConfig` 的全部配置引用并复查零残留。
- **[风险] Python import 漏改导致 import error** → 缓解：按“配置消费者 / 数据消费者”两组清单逐个更新，并做全量 import smoke check。
- **[权衡] 删除兼容壳文件会让未枚举的外部引用立即失效** → 可接受；仓库内部引用已知且可控，本次目标是建立清晰边界而非维持旧路径长期共存。
