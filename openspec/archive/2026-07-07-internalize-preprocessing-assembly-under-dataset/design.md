## Context

当前 `rkmeans_train` 已经完成了 reader factory 与 shuffle contract 的第一阶段收敛，但 preprocessing 链路仍未完全符合目标架构。具体表现为：

- preprocessing 依赖的数据仍大量经过 `dataset_config` 中转；
- `features` 到 preprocessing 所需元数据的派生主要依赖 Hydra resolver；
- preprocessing 函数签名仍普遍保留 `dataset_config` 参数；
- preprocessing 的装配更多发生在 YAML，而不是 dataset 内部。

用户已明确新的从属关系：

- `dataset_config` -> `dataset` -> `preprocessing`

因此，dataset 是 preprocessing 的装配者，preprocessing 是 dataset 的子模块。preprocessing 不允许再反向接收 `dataset_config`，否则会破坏依赖方向并形成架构层面的循环引用。

## Goals / Non-Goals

**Goals:**
- 让 `rkmeans_train` 的 preprocessing 装配职责回到 dataset 内部
- 让 preprocessing 函数不再接收 `dataset_config`
- 让 preprocessing 只接收最小必要参数
- 显著减少 YAML 中为 preprocessing 服务的大量 resolver 派生字段

**Non-Goals:**
- 不要求本次迁移所有实验的 preprocessing 体系
- 不改变 `rkmeans_train` 的业务语义、embedding lookup 结果或 collate 输出结构
- 不要求彻底消灭整个仓库中的所有 resolver，只聚焦本次链路里“仅服务 preprocessing”的派生属性

## Decisions

### D1: preprocessing 装配归 dataset 所有
- **选择**：dataset 接收 `dataset_config` 后，在模块内部把 preprocessing 声明配置装配成最终的 `list[callable]`
- **理由**：这符合 `dataset -> preprocessing` 的单向依赖关系，也能避免在 YAML 中直接堆叠过多装配细节

### D2: preprocessing 不接收 dataset_config
- **选择**：preprocessing 函数签名不得再依赖 `dataset_config` / `config`
- **理由**：preprocessing 是 dataset 的叶子能力，不应反向读取上层模块配置

### D3: preprocessing 采用最小参数原则
- **选择**：每个 preprocessing 只接收最小必要参数，例如 `field_type_map`、`embedding_bundle`、`features_to_consider` 等局部参数
- **理由**：这样函数边界最清晰，也便于测试和复用

### D4: 从 features 派生 preprocessing 依赖的逻辑收回 Python 侧
- **选择**：尽量由 dataset 内部的 assembler/helper 从 `features` 或 preprocessing 声明配置中构造 callables，而不是在 YAML 中大面积使用 resolver 派生中间属性
- **理由**：减少配置噪声，提升可读性，并让推导逻辑集中在代码中

### D5: rkmeans_train 作为首个模板，先追求可复制的内部装配模式
- **选择**：本次只把 `rkmeans_train` 跑通并定型，再以该模式迁移其他实验
- **理由**：控制改动面，同时沉淀清晰模板

## Risks / Trade-offs

- **[风险] dataset 内部 assembler 过重，可能把 dataset 变成“隐式配置解释器”**  
  **缓解**：将 assembler 逻辑限制在 preprocessing 装配，必要时拆到 dataset 子模块 helper，而不是塞进 `__iter__()`。

- **[风险] 一次性移除 `dataset_config` 参数会触达多个 preprocessing 函数**  
  **缓解**：优先处理 `rkmeans_train` 当前实际使用的几步 preprocessing，避免无关扩散。

- **[权衡] 把 resolver 逻辑迁到 Python 侧会减少 YAML 声明性**  
  **可接受**：用户当前优先级是可读性、边界清晰和后续迁移成本，而不是最大化 YAML 内联声明。

## Migration Plan

1. 识别 `rkmeans_train` 当前 preprocessing 所需的最小参数集合
2. 设计 dataset_config 中的 preprocessing 声明配置结构
3. 在 dataset 模块内部引入 preprocessing assembler/helper
4. 更新 preprocessing 函数签名，移除 `dataset_config` 依赖
5. 清理 `rkmeans_train.yaml` 中仅服务 preprocessing 的 resolver 派生字段
6. 做最小 compose / instantiate / 运行链路验证，确认该模式可作为后续模板
