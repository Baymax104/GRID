## 1. Config Cleanup

- [x] 1.1 Remove `initialize_on_cpu` from `configs/model/rkmeans_train.yaml`.
- [x] 1.2 Remove `initialize_on_cpu` from `configs/model/rkmeans_inference.yaml`.
- [x] 1.3 Remove `initialize_on_cpu` from `configs/model/rvq_train.yaml`.
- [x] 1.4 Remove `initialize_on_cpu` from `configs/model/rqvae_train.yaml`.

## 2. RKMeans Implementation

- [x] 2.1 Remove `initialize_on_cpu` from `ResidualKMeans.__init__()` and stop storing/passing it.
- [x] 2.2 Remove `initialize_on_cpu` from `KMeansLayer.__init__()` and its attributes.
- [x] 2.3 Update RKMeans K-Means++ helper/call sites so initialization always uses the current buffer device.
- [x] 2.4 Confirm RKMeans distributed rank-zero broadcast initialization still works without the CPU initialization parameter.

## 3. RVQ / RQVAE Implementation

- [x] 3.1 Remove `initialize_on_cpu` from `ResidualVectorQuantization.__init__()` and its attributes.
- [x] 3.2 Update RVQ K-Means++ initialization helper/call sites to remove the CPU branch.
- [x] 3.3 Remove `initialize_on_cpu` from `ResidualQuantizationVAE.__init__()` and its attributes.
- [x] 3.4 Update RQVAE K-Means++ initialization helper/call sites to remove the CPU branch.

## 4. Tests and Verification

- [x] 4.1 Update RKMeans unit tests to construct `KMeansLayer` / `ResidualKMeans` without `initialize_on_cpu`.
- [x] 4.2 Add or update assertions that `initialize_on_cpu` is absent from quantization model config and constructor signatures.
- [x] 4.3 Run targeted unit tests for touched quantization code.
- [x] 4.4 Run Hydra target/import smoke check for RKMeans, RVQ, and RQVAE model targets.
- [x] 4.5 Run a final repository search confirming no active code/config references to `initialize_on_cpu` remain outside archived OpenSpec history.
