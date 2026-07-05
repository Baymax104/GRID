## Context

当前 `configs/experiment/*.yaml` 中普遍存在同一种结构问题：`data_loading`、`model` 等区域同时混合纯参数、共享参数以及带 `_target_` 的主要装配子树。由于 Hydra 默认递归 instantiate，这些嵌套 `_target_` 节点既是配置内容，又是对象装配边界，导致 experiment 文件很难一眼看清“主链路由哪些主要组件组成”。

用户已经明确本次不做 repo 级共享配置，也不做跨 experiment 共享配置；重构应当限制在单个 experiment 文件内部，通过引入 `components:` 区来提升主链路可读性。`data_loading` 与 `model` 将作为参数域名称保留，主要表达纯参数、共享值和组件引用，而不是继续承载大块主要组件定义。

## Goals / Non-Goals

**Goals:**
- 对所有 official experiment 配置实施单文件内组件化重组。
- 为每个 experiment 引入显式 `components:` 区，承载主要 instantiate 子树。
- 让 `data_loading`、`model` 等成为参数域名称，尽量只保留纯参数、共享值和对组件的引用。
- 保持 experiment 对 `launcher_utils.py` 当前装配入口的兼容，不改变 Python 侧顶层 instantiate 行为。
- 让主链路更容易阅读：打开 experiment 文件即可快速定位 datamodule、主要 dataloader/dataset、核心模型子模块等组成部分。

**Non-Goals:**
- 不引入 `configs/components/` 这类 repo 级共享配置目录。
- 不做跨 experiment 去重或共享模板抽取。
- 不系统性改造 `callbacks/logger/trainer/paths/extras/hydra` 默认配置层，只做必要适配。
- 不引入 `_recursive_: false` 或新的 Python 侧配置解析逻辑。

## Decisions

### 1. 每个 experiment 文件内新增 `components:` 区
- 决策：所有 official experiment 统一引入 `components:` 顶层区块，作为主要 instantiate 子树的容器。
- 该区内组件按 experiment 自身需要命名，例如：
  - `components.data_loading.dataset`
  - `components.data_loading.train_dataloader`
  - `components.model.quantization_layer`
  - `components.model.loss`
- 原因：在不跨文件共享的前提下，最直接地把“主要组件定义”和“参数域”分离。
- 备选方案：把组件定义继续留在 `data_loading` / `model` 深层。未采用，因为仍会维持当前主链路混杂问题。

### 2. `data_loading` 与 `model` 作为参数域保留
- 决策：experiment 继续保留 `data_loading:` 与 `model:` 顶层名称，但其职责调整为参数域：
  - 存放纯参数、共享值、映射、路径引用
  - 通过插值引用 `components:` 中的主要装配节点
- 原因：保留当前领域语义，避免彻底重命名带来的额外适配成本。

### 3. 只抽“主要组件”，允许保留局部 inline `_target_`
- 决策：不是所有 `_target_` 都必须提升到 `components:`。以下节点优先抽出：
  - datamodule / dataloader / dataset / collate / label transform
  - 量化层、初始化器、调度器、loss、evaluator、主要 HF 子模块
- 允许保留 inline 的节点：
  - 很短、只出现一次、强绑定局部语义的小组件
  - 为了抽出反而会让引用更绕的小型 `_target_` 节点
- 原因：避免为“全部节点组件化”付出过高复杂度。

### 4. 以 experiment 层改动为主，默认层仅做必要适配
- 决策：主要工作集中在 `configs/experiment/*.yaml`。`configs/main.yaml` 和默认层配置只有在引用关系需要时才做最小适配。
- 原因：用户已经明确本次重点是 experiment 可读性，而不是整个 Hydra 配置体系重构。

## Risks / Trade-offs

- [每个 experiment 文件会变长] → 接受单文件内部长度上升，以换取主链路层次更清晰。
- [不做跨 experiment 共享会保留重复] → 本轮优先解决可读性，不追求去重最优。
- [组件命名不稳定会造成不同 experiment 风格漂移] → 实施时需定义统一命名约定，例如 `components.data_loading.*`、`components.model.*`。
- [过度抽取导致跳转反而增多] → 明确只抽主要组件，保留少量局部 inline `_target_`。

## Migration Plan

1. 先定义 experiment 内统一骨架与组件命名规范。
2. 先改造代表性的 item 类 experiment 与 sequence 类 experiment，验证骨架可行。
3. 再将同一骨架推广到所有 remaining official experiments。
4. 做配置级 smoke check，确认 launcher 顶层 instantiate 入口仍可解析各 experiment。

## Open Questions

- 当前无阻塞性开放问题。
