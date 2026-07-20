## Context

当前 official experiment 配置已经完成单文件内组件化：主要装配子树被提升到 `components:`，`data_loading` 与 `model` 更接近参数域。但 Python 侧顶层实例化入口仍未完全对齐这套结构。`launcher_utils.py` 现在仍直接读取 `cfg.data_loading.datamodule`、`cfg.model`、`cfg.trainer`、`cfg.callbacks`、`cfg.logger`，其中一部分只是参数域到 `components` 的转发别名。

用户已明确希望进一步收紧依赖关系，规范为：`python -> components config -> argument config`。同时，本轮不仅要处理 datamodule 和 model，还要把 callbacks/logger/trainer 一并纳入 `components`。其余设计采用默认建议：`components.model.root`、`components.trainer.root`、`components.callbacks`、`components.logger`。

## Goals / Non-Goals

**Goals:**
- 让 Python 侧直接实例化入口统一指向 `components` 下的装配根。
- 删除参数域中仅作为装配别名存在的中转节点，例如 `data_loading.datamodule`。
- 为 official experiment 统一组件入口命名：
  - `components.data_loading.datamodule`
  - `components.model.root`
  - `components.trainer.root`
  - `components.callbacks`
  - `components.logger`
- 保持 `data_loading`、`model` 等参数域仅表达参数和值引用。

**Non-Goals:**
- 不重新设计 Hydra 递归实例化机制。
- 不引入 repo 级共享配置。
- 不改变具体业务组件的行为，只规范配置入口依赖关系。
- 不要求本轮就重构 `log_hyperparameters()` 为完整记录所有 component 装配细节。

## Decisions

### 1. Python 顶层实例化统一改为读取 `components`
- 决策：`launcher_utils.py` 中 datamodule、model、trainer 的 `hydra.utils.instantiate(...)` 入口改为直接读取 `cfg.components.*`。
- callbacks/logger 的集合实例化也改为从 `cfg.components.callbacks` / `cfg.components.logger` 读取。
- 原因：彻底移除参数域中的装配根别名，使 Python 依赖关系与配置分层一致。

### 2. 参数域不再承担装配根中转
- 决策：从 experiment 配置中删除或收缩如下中转入口：
  - `data_loading.datamodule`
  - `model` 顶层的装配根职责
  - `trainer` / `callbacks` / `logger` 顶层的直接实例化职责
- 对 `model`，统一提升为 `components.model.root`，参数域 `model` 仅保留纯参数和值。
- 原因：参数域只应承载参数，而不是 Python 直接实例化根。

### 3. callbacks/logger 直接使用集合节点
- 决策：对 callbacks 和 logger 两个集合型入口直接使用：
  - `components.callbacks`
  - `components.logger`
- 原因：`components.callbacks` / `components.logger` 本身已经具备“多项定义集合”的语义，额外包一层 `definitions` 没有提供额外信息。

### 4. trainer 与 model 使用 `root`
- 决策：单实例装配根统一命名为 `root`：
  - `components.model.root`
  - `components.trainer.root`
- 原因：与参数域内其他组件区分清楚，且表明这是 Python 直接实例化的单个顶层对象。

### 5. 日志记录默认继续以参数域为主
- 决策：`log_hyperparameters()` 默认继续优先记录 `data_loading`、`model`、`trainer` 等参数域；是否额外记录 `components` 本轮不作为硬性目标。
- 原因：这能控制 blast radius，避免在“入口规范化”变更里顺带扩大日志语义变更。

## Risks / Trade-offs

- [改动 launcher 与所有 official experiment，范围较大] → 通过统一命名和配置级 smoke check 控制风险。
- [model 参数域与 `components.model.root` 的引用关系可能更绕] → 通过明确命名规范和局部注释保持可读性。
- [callbacks/logger 迁入 components 后，默认配置层的读取路径变化] → 保持集合形状不变，仅更换 Python 读取路径。
- [日志仍主要记录参数域，可能丢失完整装配视图] → 暂时接受，后续如需要再单独增强。

## Migration Plan

1. 先修改 `launcher_utils.py` 与相关实例化辅助代码，支持从 `components` 读取入口。
2. 再统一重构 official experiment 配置的入口命名与引用路径。
3. 做配置 compose / instantiate smoke check，确认 Python 入口能正确读取新路径。
4. 复核参数域中不再残留仅用于 Python 直接实例化的别名。

## Open Questions

- 当前无阻塞性开放问题。
