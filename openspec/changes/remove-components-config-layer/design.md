## Context

当前仓库刚完成按组件类型拆分 official experiment 配置：`configs/experiment/*.yaml` 已收缩为薄入口，组件参数分别下沉到 `configs/data_loading/`、`configs/model/`、`configs/trainer/`、`configs/logger/`、`configs/callbacks/`。但为了兼容现有 Python 装配入口，这些组件文件内部仍然保留 `components.<group>` 与顶层参数域双视角。例如 `configs/model/sem_embeds_inference.yaml` 同时包含 `model:` 和 `components.model:`。用户已明确要求继续收敛：不保留参数域，去掉 `components` 容器层，但保留 `root` 以表示单实例顶层组件。

## Goals / Non-Goals

**Goals:**
- 移除配置树中的 `components` 包装层。
- 移除独立参数域，让组件配置文件直接承载组件定义。
- 保留 `root` 节点作为 model/trainer 等单实例顶层组件入口。
- 将 Python 侧实例化入口收敛到顶层组件路径。

**Non-Goals:**
- 不追求跨实验复用。
- 不去掉 `root` 节点。
- 不改变 datamodule/model/trainer/callback/logger 的业务行为。

## Decisions

### 1. Python 入口去掉 `components` 包装层
- 决策：`launcher_utils.py` 中实例化路径统一改为：
  - `cfg.data_loading.datamodule`
  - `cfg.model.root`
  - `cfg.trainer.root`
  - `cfg.callbacks`
  - `cfg.logger`
- 原因：这与新的组件文件目录结构更一致，也避免在配置树中保留额外 assembly namespace。

### 2. 组件文件直接挂到顶层 package
- 决策：组件配置文件使用更直接的 package 语义，例如：
  - `data_loading/<experiment>.yaml` → 顶层 `data_loading`
  - `model/<experiment>.yaml` → 顶层 `model`
  - `trainer/<experiment>.yaml` → 顶层 `trainer`
  - `callbacks/<experiment>.yaml` → 顶层 `callbacks`
  - `logger/<experiment>.yaml` → 顶层 `logger`
- 原因：文件名已经表达了组件类别，文件内部不需要再写 `components.<group>` 的重复层级。

### 2.1 experiment defaults 显式指定挂载位置
- 决策：`configs/experiment/*.yaml` 中使用 defaults package relocation 明确声明挂载位置，例如：
  - `/data_loading@data_loading: <experiment>`
  - `/model@model: <experiment>`
  - `override /trainer@trainer: <experiment>`
  - `override /callbacks@callbacks: <experiment>`
  - `override /logger@logger: <experiment>`
- 原因：这样 experiment 文件会直接体现每个子配置挂到哪个顶层键下，同时子配置文件本身可以进一步展平，不再写外层包装键。

### 3. 保留 `root`，移除参数域
- 决策：对单实例顶层组件保留 `root`，如 `model.root`、`trainer.root`；同时移除独立参数域视图。
- 原因：`root` 仍然清楚表达“顶层可实例化对象”，而去掉参数域可以避免双视角并进一步减薄组件文件。
- 备选方案：连 `root` 一并去掉；被否决，因为 model/trainer 下通常还有 decoder、optimizer、dataset-like 子节点，保留 `root` 更清晰。

### 3.1 子配置文件去掉外层 group key
- 决策：像 `configs/model/<experiment>.yaml`、`configs/data_loading/<experiment>.yaml` 这类文件内部不再写 `model:` / `data_loading:` 外层包装，而是直接写该 group 下的内容本体。
- 原因：目录名和 defaults 挂载位置已经表达了所属 group，再重复一层键只会增加文件噪音。

### 4. callbacks/logger 保持集合节点
- 决策：`callbacks`、`logger` 直接作为集合型顶层节点保留，不再额外包装。
- 原因：它们天然是 map 结构，不需要 `root`。

## Risks / Trade-offs

- [需同步修改 Python 装配入口] → 通过最小 blast radius 改 `launcher_utils.py` 并做 compose/instantiate smoke check。
- [去掉参数域后日志/override 语义改变] → 接受这一变化，因为用户已明确不再保留参数域。
- [data_loading 层级仍相对复杂] → 保留 `data_loading.datamodule` 作为顶层入口，其余 dataset/dataloader/collate 继续在 `data_loading` 内部分层。

## Migration Plan

1. 先调整 Python 读取路径。
2. 再将各组件配置文件收敛为顶层 package 结构，并去掉文件内部的外层 group key。
3. 收缩 `configs/experiment/*.yaml` 中与旧 `components` 相关的残留假设，同时用 defaults package relocation 显式指定挂载位置。
4. 执行 compose / instantiate smoke check，并做全文搜索确认 `cfg.components` 不再是官方依赖路径。

## Open Questions

- 当前无阻塞性开放问题。
