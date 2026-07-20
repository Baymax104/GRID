## Why

所有量化实验均启用 residual tracking，而训练与评估的诊断统计也无条件依赖每层 residual。保留可关闭但无法支持这些统计的配置开关会造成无效接口与错误的默认行为。

## What Changes

- 移除量化模型配置中的 `track_residuals: true`。
- 移除三个 residual quantizer 的 `track_residuals` 构造参数和内部状态。
- 始终收集并返回逐层 residual，使诊断统计的输入保持确定可用。
- **BREAKING**：不再支持通过 Hydra 或直接构造模型关闭 residual tracking。

## Capabilities

### New Capabilities
- `unconditional-residual-tracking`: 定义 residual quantizer 始终产生逐层 residual 诊断数据且不暴露关闭开关的契约。

### Modified Capabilities

无。

## Impact

- `configs/model/{rkmeans_train,rkmeans_inference,rvq_train,rqvae_train}.yaml`
- `src/quantization/residual_kmeans.py`
- `src/quantization/residual_vector_quantization.py`
- `src/quantization/residual_quantization_vae.py`
- 量化模型的 Hydra 覆盖接口与直接 Python 构造接口。
