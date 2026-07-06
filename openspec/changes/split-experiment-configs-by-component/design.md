## Context

当前 official experiment 文件已经完成了 `components` 入口规范化，但大多数 experiment 仍然把 data loading、model、trainer、logger、callbacks 的具体参数与装配关系写在同一个 YAML 文件内。像 `trainer.root`、`model.root`、各类 dataloader config 中存在大量“把参数域值再插回实例化配置”的映射，导致加删构造参数时经常需要改两个位置。用户已明确提出新的组织方向：不追求跨实验复用，而是按组件类型拆目录，例如 `trainer/<experiment>.yaml`、`model/<experiment>.yaml` 等，由 `experiment/<experiment>.yaml` 统一导入并做装配。

## Goals / Non-Goals

**Goals:**
- 将 official experiment 重组为“experiment 薄入口 + per-component config group”的结构。
- 让单个实验的 trainer/model/logger/callbacks/data_loading 参数主要只在各自组件文件中维护一份。
- 保持 Python 侧 `cfg.components.*` 实例化入口稳定。
- 让 experiment 文件更像总装配索引，并清晰体现跨组件依赖。

**Non-Goals:**
- 不追求不同实验之间的配置复用。
- 不重写 Hydra 装配机制。
- 不要求本轮顺带重构所有日志语义或 CLI 约定。

## Decisions

### 1. 按组件类型建立 experiment-specific config groups
- 决策：新增或扩展 `configs/trainer/`、`configs/model/`、`configs/data_loading/`、`configs/logger/`、`configs/callbacks/`，每个 official experiment 在这些目录下有对应文件。
- 原因：这样单个组件的参数可以集中维护，同时继续保留 experiment 级别的独立性。
- 备选方案：保留单文件 experiment + fragments 复挂；被否决，因为用户更关注主文件可读性，不希望保留大量隐藏展开关系。

### 2. experiment 文件收缩为总装配入口
- 决策：`configs/experiment/<name>.yaml` 主要保留：
  - defaults 导入组件配置
  - 运行元信息与少量人工输入
  - 必要的顶层装配关系/跨组件引用
- 原因：让 experiment 文件从“大型参数仓库”变回“实验入口索引页”。

### 3. 组件文件直接承担大部分实例化配置
- 决策：像 `trainer/<experiment>.yaml`、`logger/<experiment>.yaml` 这类文件，默认直接提供 `components.*` 下对应节点所需的大部分最终配置，而不是先放一份纯参数再由 experiment 手工映射。
- 原因：这最直接地消除双处维护。
- 备选方案：保留顶层纯参数域并让 component 文件只放参数；被否决，因为会重新引入两处视图和更长的配置树。

### 4. data_loading 按 experiment 整体下沉，不再过度细分
- 决策：`data_loading/<experiment>.yaml` 作为该实验 data loading 相关配置的主要承载文件，内部可以继续包含 dataset/dataloader/collate/preprocessing/datamodule 等子树。
- 原因：data loading 是最复杂的部分，但如果继续拆得过细，会让 defaults 关系过于零碎。

## Risks / Trade-offs

- [文件数量增加] → 以更短、更清晰的 experiment 主文件换取更多组件文件，属于预期 trade-off。
- [某些跨组件依赖可能分散] → 在 experiment 文件中仅保留必要的总装配关系，并在组件文件命名上保持 experiment 对齐。
- [data_loading 仍然较大] → 接受其作为单独复杂组件文件存在，避免过早进一步细分。
- [迁移范围覆盖全部 official experiments] → 通过按 item 类 / sequence 类分批迁移和 compose smoke check 控制风险。

## Migration Plan

1. 先确定组件目录和命名规范。
2. 先迁移一个 item 类 experiment 和一个 sequence 类 experiment 验证结构。
3. 批量迁移其余 official experiments。
4. 执行 compose / instantiate smoke check，确认 Python 入口仍然读取 `cfg.components`。

## Open Questions

- 当前无阻塞性开放问题。
