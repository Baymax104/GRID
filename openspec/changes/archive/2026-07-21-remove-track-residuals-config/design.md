## Context

`ResidualKMeans`、`ResidualVectorQuantization` 与 `ResidualQuantizationVAE` 都通过 `track_residuals` 控制是否累积每层 residual。所有现有量化配置固定传入 `true`，而训练和评估统计无条件使用该输出；关闭开关会使统计路径接收 `None`。

## Goals / Non-Goals

**Goals:**
- 使三个 residual quantizer 始终生成形状为 `(batch_size, n_features, n_layers)` 的逐层 residual tensor。
- 移除不可用的配置和构造接口，消除依赖该标志的条件分支。
- 保持现有默认实验的模型输出和诊断指标不变。

**Non-Goals:**
- 不改变 residual 的计算顺序、量化算法或诊断指标公式。
- 不修改非量化实验的模型配置。
- 不为停用 residual tracking 提供替代开关。

## Decisions

### 将 residual tracking 设为量化模型的固定行为

三个 `forward()` 在每层从当前 residual 减去该层 embedding 后，无条件追加该 tensor，并在循环结束后无条件堆叠。这样与既有 `true` 行为完全一致，并保证下游统计始终获得 tensor。

备选方案是保留构造参数并将默认值改为 `true`。不采用该方案，因为仍会暴露一个已知无法安全关闭的分支，也无法移除四处冗余 YAML。

### 完整移除公共接口而非仅忽略配置值

从三个构造函数和实例状态删除 `track_residuals`，并从四个量化模型 YAML 删除该键。这使 Hydra 无法再将该参数注入模型，且直接构造模型时不再支持该选项。

### 保持输出元组和 residual tensor 布局

不改变 `forward()`/`model_step()` 的元组位置、tensor dtype 或 `(batch_size, n_features, n_layers)` 布局，避免影响训练、评估和预测调用者。

## Risks / Trade-offs

- **无法节省 residual 收集的内存** → 这是已有所有量化实验的固定开销；删除无效开关优先于保留不可支持的优化路径。
- **旧命令行 Hydra override 失败** → 这是明确的破坏性接口移除；在 proposal 和 spec 中记录，并通过配置组合验证发现遗留用法。
- **遗漏一个模型实现导致行为不一致** → 对三个 residual quantizer 使用同一检查：构造签名、配置键、残差追加和 stack 分支均不再出现该标志。
