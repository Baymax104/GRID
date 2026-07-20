## Context

当前 `rkmeans_train` 的 preprocessing 已具备两个重要性质：

- preprocessing 为 row-only
- preprocessing 函数不再接收 `dataset_config`

但装配方式走向了 dataset 内部 assembler / registry：

- `dataset_config.features`
- `dataset_config.preprocessing_steps`
- `preprocessing_assembly.py`

这套方式虽然扩展性更强，但用户明确希望当前场景以“可读性优先”为第一目标，因为 preprocessing 通常只在某个实验中声明一次。既然复用需求不强，过度抽象会降低理解效率。

因此，本次需要回到“配置文件直接声明 preprocessing”的方式，同时保留新接口的纯函数约束。

## Goals / Non-Goals

**Goals:**
- 让 `rkmeans_train.yaml` 中一眼可见 preprocessing chain 及其参数
- 删除 assembler / `preprocessing_steps` / `features` 这条仅服务 preprocessing 的中间层
- 保持 preprocessing 接口纯函数化与最小参数化

**Non-Goals:**
- 不回退到 `dataset_config` 传入 preprocessing 的旧模式
- 不恢复基于 Hydra resolver 的 preprocessing 参数派生
- 不要求本次同时重构其他实验

## Decisions

### D1: preprocessing chain 直接在配置中声明
- **选择**：在 `rkmeans_train.yaml` 中显式声明 `preprocessing_functions`，每步用 `_partial_` 和最小参数配置
- **理由**：可读性最好，最符合当前用户优先级

### D2: dataset 不再推导 preprocessing 参数
- **选择**：dataset 只消费 `dataset_config.preprocessing_functions`
- **理由**：dataset 应回到更薄的一层，只负责执行 chain，不负责解释 preprocessing schema

### D3: 删除仅服务 assembler 的中间层
- **选择**：移除 `preprocessing_assembly.py`，并从相关 dataset config 中去掉 `preprocessing_steps`、`features`（若只服务该路径）
- **理由**：既然新目标是配置直写，就不再保留多余中间层

### D4: 参数必须显式可读，不再通过 resolver 派生
- **选择**：在 YAML 中显式写 `features_to_consider`、`field_type_map`、`embedding_bundle` 等参数或局部加载配置
- **理由**：这比 resolver 链更直观，也符合“读配置即知预处理链路”的目标

## Risks / Trade-offs

- **[风险] 配置文件会变长、存在部分重复**  
  **缓解**：用户已明确可读性优先，这一代价可接受。

- **[风险] 如果未来多个实验大量复用相同 preprocessing，配置直写可能显得啰嗦**  
  **缓解**：当前只针对 `rkmeans_train`，未来若复用性变强，可再抽象。

## Migration Plan

1. 在 `rkmeans_train.yaml` 中显式写回 `preprocessing_functions`
2. 用字面量/局部加载配置替代 assembler 派生参数
3. 更新 dataset 直接消费 `preprocessing_functions`
4. 删除 `preprocessing_assembly.py` 与相关中间字段
5. 做最小 compose / import / row-chain 验证
