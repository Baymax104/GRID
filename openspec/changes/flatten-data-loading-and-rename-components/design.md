## Context

当前数据加载源码集中在 `src/data/loading/` 下，包含 `components/`（8 个模块）、`datamodules/`（4 个模块）、`utils.py`。`loading` 这一层中间目录是历史遗留，与 `src/data/` 语义重复。此前配置层已做过 `configs/data_loading/` → `configs/data/` 的更名（见已实施变更 `rename-data-loading-and-prune-unused-defaults`），但源码层 `src/data/loading/` 尚未跟进扁平化。

`components/` 各模块命名参差：`interfaces.py` 名不副实（内容是配置类与数据容器而非接口）、`dataloading.py` 与上层目录语义重复（实为 dataset 定义）、`custom_dataloader.py` 语义模糊、`label_function.py` 单数与其他复数文件名不一致、`pre_processing.py` 下划线分割不符合惯例。

引用面已完整枚举：7 个 `configs/data/*.yaml` 的 `_target_`、6 个外部 Python 模块（仅引用 `components.interfaces`）、9 处 `src/data/loading/` 内部相互 import、2 处文档（`AGENTS.md`、`src/data/README.md`）。仓库无测试套件，验证依赖 import smoke check 与 Hydra compose 检查。

## Goals / Non-Goals

**Goals:**
- 消除 `loading/` 中间层，使 `src/data/` 直接承载 `components/`、`datamodules/`、`utils.py`
- components 模块文件名准确反映内容职责，命名风格统一（复数、去下划线、职责明确）
- 核心类 `UnboundedSequenceIterable` 重命名为简洁的 `SequenceDataset`
- 全部引用同步更新，保持运行时行为与外部实验参数语义不变

**Non-Goals:**
- 不拆分 `data_models.py`（原 `interfaces.py`）为 `configs.py` + `data_containers.py`——改动面与收益不匹配
- 不改变任何模块的内部逻辑、函数签名、配置参数语义
- 原始文件数据读取组件命名后续收敛为 `readers.py` 更贴近职责
- 不处理 `src/data/exploration/` 下的 ipynb（无 loading 引用）
- 不回改已实施变更 `extract-datamodule-base-and-split-files` 的 spec delta

## Decisions

### D1: 直接消除 `loading/` 中间层，不保留兼容别名
- **选择**：物理移动文件，模块路径从 `src.data.loading.*` 收敛为 `src.data.*`，不保留旧路径转发。
- **理由**：仓库内引用全部可控（已枚举 22 处），无外部消费者；保留别名会延长过渡期并增加维护面。Hydra `_target_` 按字符串解析，re-export 无法让旧 `_target_` 路径继续工作。
- **备选**：在 `src/data/loading/__init__.py` 用 re-export 转发——否决，因配置 `_target_` 路径无法靠 re-export 兼容，且会让旧路径长期残留。

### D2: `interfaces.py` → `data_models.py`（不拆分，不用 models/schemas）
- **选择**：整体重命名为 `data_models.py`，不拆分。
- **理由**：内容是"数据配置类 + 批数据容器 dataclass"，概念上同属"数据模型"（data engineering 通用术语）。`models.py` 会被误读为神经网络模型（项目里 `model` 专指 LightningModule，见 `cfg.model.root`）；`schemas.py` 偏向 API 校验语境。拆分为 `configs.py` + `data_containers.py` 会增加改动面（6 处外部引用需重定向到两个文件），收益不匹配。
- **备选**：`models.py`（语义冲突）、`schemas.py`（语境不符）、拆分（改动面过大）——均否决。

### D3: `UnboundedSequenceIterable` → `SequenceDataset`，文件 `dataloading.py` → `datasets.py`
- **选择**：类重命名为 `SequenceDataset`，文件改为 `datasets.py`。
- **理由**：类继承 `BaseDataset` + `IterableDataset`，是唯一的序列数据集实现；`UnboundedSequenceIterable` 后缀 `Iterable` 与基类 `IterableDataset` 语义重复且冗长。`SequenceDataset` 最简洁，后缀 `Dataset` 准确表达本质。
- **备选**：`IterableSequenceDataset`（冗长）、`StreamingSequenceDataset`（"流式"是引申义）——否决。

### D4: 其余模块按"复数 + 去下划线 + 职责明确"统一
- `custom_dataloader.py` → `dataloaders.py`：复数，与 `datasets.py` 对称。
- `collate_functions.py` → `collate.py`：简洁，内容即 collate 函数集。
- `label_function.py` → `label_functions.py`：单数改复数，与其他复数文件名一致。
- `pre_processing.py` → `preprocessing.py`：去下划线，符合 Python 惯例（如 `sklearn.preprocessing`）。
- `readers.py`：用于承载原始文件数据读取器定义。

### D5: `src/data/` 补空 `__init__.py`
- **选择**：新增空 `__init__.py`。
- **理由**：`src/utils/`、`src/common/`、`src/embedding/` 等同级包都有 `__init__.py`；`src/data/` 展平后成为直接的代码包，应保持一致。虽 `rootutils` 注入 path 后命名空间包也能工作，但显式 `__init__.py` 更规范。

## Risks / Trade-offs

- **[风险] 重命名 `UnboundedSequenceIterable` 可能遗漏字符串引用** → 缓解：该类通过 `dataset_class` 字段以 `_target_` 引用，7 个 yaml 全部枚举在内；改后用 `rg` 复查全仓 `UnboundedSequenceIterable` 零残留。
- **[风险] 修改未归档的 `datamodule-structure-alignment` capability 路径 requirement** → 缓解：以 `MODIFIED` delta 精确替换"Datamodule module paths SHALL reflect separated roles" requirement 的路径，归档时 living specs 反映最终路径。若 openspec 不接受跨未归档变更的 MODIFIED，则退化为在新 capability 中以 ADDED requirement 规定最终路径，并在归档阶段协调。
- **[权衡] 不保留旧路径兼容，未枚举的外部脚本会断** → 可接受，仓库为研究代码，引用面内部可控。
