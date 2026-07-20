## Context

当前配置结构已经去掉了 `components` 容器层，并把 official experiment 拆分到了 `configs/model/`、`configs/trainer/`、`configs/data_loading/`、`configs/logger/`、`configs/callbacks/`。同时，`configs/main.yaml` 已经不再默认导入 `trainer/default.yaml`、`logger/default.yaml`、`callbacks/default.yaml`。从官方入口视角看，这些 default 配置很可能已经无引用。另一方面，配置和代码中仍广泛使用旧命名 `data_loading`，包括 Hydra defaults 挂载路径、YAML 插值、Python 侧的 `cfg.data_loading` 读取、日志字段、warning 和 config tree 打印顺序。用户已明确要求评估 default 文件是否仍需保留，并在必要时删除；同时希望将 `data_loading` 配置统一更名为 `data`。

## Goals / Non-Goals

**Goals:**
- 删除对 official 入口无用的 trainer/logger/callbacks default 配置层。
- 将 `data_loading` 顶层配置、目录名、Python 读取路径和日志键统一更名为 `data`。
- 保持 official experiments 可 compose、可实例化，不引入额外兼容层。

**Non-Goals:**
- 不重构 data 组件内部语义（dataset/dataloader/collate 的结构保持不变，只改命名）。
- 不回写归档文档中的历史 `data_loading` 描述。
- 不改变非 official、历史遗留配置的行为约定，除非它们与当前官方入口直接冲突。

## Decisions

### 1. 删除不再被 official 入口使用的 default 组件配置
- 决策：删除 `configs/trainer/default.yaml`、`configs/logger/default.yaml`、`configs/callbacks/default.yaml`，并同步评估 `callbacks/model_checkpoint.yaml`、`callbacks/early_stopping.yaml`、`callbacks/model_summary.yaml` 是否仍有官方引用；若无，则一并清理。
- 原因：`configs/main.yaml` 已不再导入它们，official experiment 也都显式使用 experiment-specific 配置文件；继续保留只会形成噪音与误导。

### 2. 顶层命名统一从 `data_loading` 改为 `data`
- 决策：配置树、目录名和 Python 读取路径统一改为 `data`，例如：
  - `cfg.data.datamodule`
  - `configs/data/<experiment>.yaml`
  - `/data@data: <experiment>`
- 原因：当前其他顶层组件都已采用更短的单词名（`model`、`trainer`、`logger`、`callbacks`），`data` 与新的扁平结构更一致。

### 3. 同步修正日志与打印语义
- 决策：`log_hyperparameters()`、`print_warnings_for_missing_configs()`、`print_config_tree()` 等仍使用 `data_loading` 的地方统一改为 `data`。
- 原因：避免代码与配置树命名脱节，也让 CLI 输出与用户心智保持一致。

## Risks / Trade-offs

- [更名影响面较大] → 通过全文搜索 + compose/import smoke check 控制风险。
- [误删仍有引用的 default 模板文件] → 删除前先确认 official 入口、当前配置与 Python 代码都不再引用。
- [living spec 与历史 spec 命名不一致] → 只更新当前活跃 spec/living spec；归档文档保留历史上下文。

## Migration Plan

1. 先做全文搜索确认 default 文件与模板文件的引用状态。
2. 再完成 `configs/data_loading/` → `configs/data/`、defaults 挂载路径和 YAML 插值更名。
3. 同步修改 Python 读取路径、日志/打印辅助代码。
4. 执行 compose/import smoke check，并全文确认 `data_loading` 在官方入口链路中不再残留。

## Open Questions

- 当前无阻塞性开放问题。
