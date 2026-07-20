## Context

当前 `src/data/loading/datamodules/sequence_datamodule.py` 同时承载 `SequenceDataModule` 与 `ItemDataModule`。两者共享同一套 stage 配置、文件分配、dataset 初始化与 `DataloaderWithIterationRetry` 组装骨架，但 `ItemDataModule` 通过重写整段 `get_dataloader()` 来绕开 `SequenceDataModule` 的序列 collate 绑定逻辑。这种结构让“共享骨架”和“序列专属行为”混在同一个类层次里，导致类型关系与实际职责不一致。

本次变更已经明确约束：
- 抽取真正中性的公共基类。
- `SequenceDataModule` 与 `ItemDataModule` 改为兄弟类。
- 文件结构拆分为 `datamodules/base.py`、`datamodules/sequence.py`、`datamodules/item.py`。
- 不保留旧兼容层，所有 Hydra `_target_` 直接迁移到新模块路径。

## Goals / Non-Goals

**Goals:**
- 将 datamodule 共享实现集中到 `BaseFileDataModule`。
- 让 `SequenceDataModule` 仅承载序列任务专属 dataloader 行为。
- 让 `ItemDataModule` 仅承载 item 任务专属 dataloader 行为。
- 将 datamodule 文件布局与类型职责对齐，降低后续理解和修改成本。
- 保持现有 experiment 的运行语义不变，仅迁移实现与 `_target_` 路径。

**Non-Goals:**
- 不重构 dataset、iterator、collate function 或 dataloader 实现本身。
- 不修改 experiment 参数结构、stage 名称或 batch 语义。
- 不为旧模块路径提供临时转发或兼容导出。

## Decisions

### 1. 引入中性公共基类 `BaseFileDataModule`
- 决策：新增 `src/data/loading/datamodules/base.py`，定义 `BaseFileDataModule`。
- 该基类承载以下公共能力：
  - `__init__` 中的 `stage_to_config` / `stage_to_file_map` 管理
  - `setup()` 中的文件扫描与 `assign_files_to_workers(...)`
  - dataset 实例化、文件绑定、distributed 参数绑定
  - `train_dataloader()` / `val_dataloader()` / `test_dataloader()` / `predict_dataloader()`
  - 公共 `get_dataloader()` 模板骨架
- 原因：共享的是真正的“按文件分配并组装 dataloader”能力，而不是 sequence 语义。
- 备选方案：保留现有继承关系，仅抽 helper 函数。未采用，因为仍会保留误导性的类型层次。

### 2. 用模板方法承载子类差异
- 决策：`BaseFileDataModule.get_dataloader()` 负责完整公共流程，并通过 hook 暴露差异点：
  - `_validate_stage_config(stage, curr_config)`
  - `_build_collate_fn(curr_config)`
- `SequenceDataModule` 负责构造带 `labels/sequence_length/masking_token/padding_token/oov_token` 的 partial collate。
- `ItemDataModule` 负责直接返回 `curr_config.collate_fn`，并保留 `assign_all_files_per_worker` 的 stage 约束。
- 原因：两类 datamodule 的主要差异集中在 collate 组装与局部约束，不值得复制整段 `get_dataloader()`。
- 备选方案：让子类各自实现完整 `get_dataloader()`。未采用，因为会继续制造大段重复代码。

### 3. 文件结构按角色拆分
- 决策：拆分为三个文件：
  - `datamodules/base.py`
  - `datamodules/sequence.py`
  - `datamodules/item.py`
- 原因：让文件边界直接表达角色；文件名去除 `datamodule` 后缀，避免目录名与文件名语义重复。
- 备选方案：保留单文件，仅调整类关系。未采用，因为无法解决“实现结构可读性差”的问题。

### 4. 直接迁移引用，不保留兼容层
- 决策：所有实验配置与导出路径一次性迁移到新文件路径，不在 `sequence.py` 中转发 `ItemDataModule`。
- 原因：引用范围完全在仓库内，可控且更干净；避免长期保留历史路径。
- 备选方案：增加兼容导入层。未采用，因为会延长过渡状态并弱化重构收益。

## Risks / Trade-offs

- [Hydra `_target_` 路径遗漏] → 全量更新 `configs/experiment/*.yaml` 并做全文搜索验证。
- [抽基类时误把 sequence 专属字段提升到公共层] → 只在 `SequenceDataModule` 中访问 `labels/sequence_length/masking_token/padding_token/oov_token`。
- [重构后行为意外漂移] → 限定本轮为结构性重排，不改 dataset、collate、batch 语义，并做最小 smoke check。
- [datamodules 导出路径变化影响内部导入] → 同步更新 `datamodules/__init__.py` 与 repo 内部引用。

## Migration Plan

1. 新增 `base.py`，提取公共 datamodule 骨架。
2. 新建 `sequence.py` 与 `item.py`，分别落定两个兄弟 datamodule。
3. 删除旧的跨语义继承关系，并迁移 `datamodules/__init__.py` 导出。
4. 更新所有 experiment YAML 的 `_target_` 到新模块路径。
5. 做全文搜索与最小导入/配置 smoke check，确认无旧 datamodule 路径残留。

## Open Questions

- 当前无阻塞性开放问题。
