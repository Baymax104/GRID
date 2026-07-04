## Context

当前 `configs/` 目录中同时存在主配置链路（`train.yaml`、`inference.yaml`、`callbacks/default.yaml`、`logger/default.yaml`、`trainer/default.yaml` 等）和若干已经不再接入 defaults、experiment、脚本或文档的残留模板。精确探查后，已确认 `configs/callbacks/local_pickle_writer.yaml`、`configs/callbacks/rich_progress_bar.yaml`、`configs/trainer/ddp.yaml` 处于未引用状态，`configs/local/` 仅保留了一个未被使用的 `.gitkeep` 占位。

本轮目标不是重构配置体系，而是做最小、低风险的噪音清理，删除已经确认未接入的死配置，为后续进一步瘦身 `configs/` 打基础。

## Goals / Non-Goals

**Goals:**
- 删除已确认未被引用的配置模板文件。
- 删除空的 `configs/local/` 目录占位。
- 保持主配置装配行为不变。

**Non-Goals:**
- 不重构 experiment 中的大段内联 `trainer` / `paths` / `logger` / `callbacks`。
- 不修改 OpenSpec 历史文档。
- 不清理“重复但仍生效”的配置层。

## Decisions

### 1. 仅删除已证实无引用的配置文件
- 决策：本轮只处理 `local_pickle_writer.yaml`、`rich_progress_bar.yaml`、`trainer/ddp.yaml` 与 `configs/local/`。
- 原因：这些文件已经通过全文搜索确认不被 defaults、experiment、脚本或文档接入。
- 备选方案：同时处理 `model_checkpoint.yaml` / `early_stopping.yaml` / `model_summary.yaml`。未采用，因为这些仍属于有效模板层的一部分。

### 2. 不顺手回收重复配置
- 决策：即使 `paths:`、`trainer:`、`logger:` 在 experiment 中存在明显重复，本轮不动。
- 原因：避免把“死配置删除”与“结构收敛重构”混在同一个变更里，提高可审查性。

## Risks / Trade-offs

- [实际存在仓库外的人工使用习惯] → 已基于仓库内 defaults / 脚本 / 文档做精确检索；如用户私有命令依赖这些文件，需要在变更说明中提醒。
- [删除空目录影响后续放本地覆盖配置] → `configs/local/` 当前无内容且无引用，后续如需要可再创建。

## Migration Plan

1. 删除三个未引用的 YAML 配置文件。
2. 删除 `configs/local/.gitkeep`，从而去掉空目录占位。
3. 做全文搜索，确认无悬挂引用。

## Open Questions

- 当前无阻塞性开放问题。
