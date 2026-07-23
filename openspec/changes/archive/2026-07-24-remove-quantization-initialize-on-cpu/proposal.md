## Why

`initialize_on_cpu` 曾用于在量化模型初始化 K-Means++ centroids 时临时把样本 buffer 移到 CPU，以降低 GPU 初始化阶段的显存压力。但当前所有可执行 model 配置都显式设置为 `false`，代码默认值也为 `false`，实际 pipeline 从不启用该路径。

保留这个未使用开关会让 RKMeans / RVQ / RQVAE 的初始化接口更难理解，并在配置中暴露一个用户不需要选择的实现细节。现在应将它从量化模型 public config 和初始化实现中移除。

## What Changes

- **BREAKING**: 从 RKMeans、RVQ、RQVAE 的 model constructor 和 Hydra model 配置中移除 `initialize_on_cpu` 字段。
- RKMeans 的 `KMeansLayer` 不再接收或传递 `initialize_on_cpu`，K-Means++ 初始化始终在当前 tensor device 上执行。
- RVQ / RQVAE 内联初始化逻辑不再支持 CPU 初始化分支，保持当前实际运行路径不变。
- 更新测试与 import/target smoke check，确保量化模型配置仍可实例化或解析。

## Capabilities

### New Capabilities

- `quantization-initialize-on-cpu-removal`: 约束量化模型不再暴露或实现 `initialize_on_cpu` 初始化开关。

### Modified Capabilities

- `config-manual-input-clarity`: 量化 model 配置不得暴露当前 pipeline 不需要手动选择的 CPU 初始化开关。
- `independent-quantization-models`: 各量化模型继续保持自包含，但不再包含未使用的 CPU 初始化路径。

## Impact

- Affected code:
  - `src/quantization/rkmeans/residual_kmeans.py`
  - `src/quantization/rkmeans/kmeans_layer.py`
  - `src/quantization/rvq/residual_vector_quantization.py`
  - `src/quantization/rqvae/residual_quantization_vae.py`
- Affected config:
  - `configs/model/rkmeans_train.yaml`
  - `configs/model/rkmeans_inference.yaml`
  - `configs/model/rvq_train.yaml`
  - `configs/model/rqvae_train.yaml`
- Existing configs or overrides that still pass `initialize_on_cpu` will fail after this change and must remove the field.
- No output artifact format, training command shape, semantic ID bundle format, or checkpoint centroid tensor layout is intended to change.
