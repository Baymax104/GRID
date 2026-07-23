# Quantization Initialize on CPU Removal

## Purpose

定义移除 quantization 模型中未使用的 `initialize_on_cpu` 配置开关与 CPU 初始化分支的协议。

## Requirements

### Requirement: Quantization models SHALL not expose initialize_on_cpu
RKMeans、RVQ、RQVAE model constructors and Hydra model configs SHALL NOT expose an `initialize_on_cpu` parameter or field.

#### Scenario: Maintainer checks quantization model configs
- **WHEN** 维护者查看 `configs/model/rkmeans_train.yaml`、`configs/model/rkmeans_inference.yaml`、`configs/model/rvq_train.yaml`、`configs/model/rqvae_train.yaml`
- **THEN** these files MUST NOT contain `initialize_on_cpu`

#### Scenario: Maintainer checks quantization model constructors
- **WHEN** 维护者查看 `ResidualKMeans`、`KMeansLayer`、`ResidualVectorQuantization`、`ResidualQuantizationVAE` 的初始化接口
- **THEN** these constructors MUST NOT accept or store `initialize_on_cpu`

### Requirement: K-Means initialization SHALL run on the current tensor device
Quantization K-Means++ initialization SHALL use the device of the current initialization buffer and MUST NOT branch through a CPU initialization option.

#### Scenario: RKMeans initializes centroids
- **WHEN** RKMeans `KMeansLayer` has collected enough residual points to initialize centroids
- **THEN** K-Means++ initialization MUST use the current buffer device
- **AND** no `initialize_on_cpu` argument may be passed through the RKMeans initialization path

#### Scenario: RVQ and RQVAE initialize centroids
- **WHEN** RVQ or RQVAE performs K-Means++ centroid initialization
- **THEN** initialization MUST use the current buffer device
- **AND** no CPU-toggle branch may exist in the initialization helper or call site
