## Why

当前 RQ-VAE 的训练曲线不可直接判断量化质量：`train/loss` 是 quantization loss 与 reconstruction loss 的加权总和，而 `val/loss` 只表示 encoded 空间的 quantization loss。与此同时，RQ-VAE 的 residual ratio / mse 使用 encoded residual 却除以原始 embedding norm，导致这些指标既不能和自身 reconstruction 目标对齐，也不能与 RKMeans/RVQ 横向比较。

## What Changes

- 让 RQ-VAE validation/test payload 返回与 train 对齐的 `loss`、`quantization_loss`、`reconstruction_loss`。
- 让 RQ-VAE `loss` 在 train/validation/test 中都表示同一个加权 total loss 语义。
- 为 RQ-VAE 拆分 encoded-space residual 指标和 reconstruction-space 指标，避免把 encoded residual 与原始 embedding norm 混合解释。
- 更新 RQ-VAE train config 的 val/test metric 声明，使 W&B 曲线可以同时观察 total、quantization、reconstruction 和空间明确的辅助指标。
- 增加聚焦 CPU 单测，锁住 RQ-VAE eval payload 和指标空间语义。
- 不在本次变更中引入 reconstruction warmup、逐层 codebook 训练阶段机或 joint finetune 策略。

## Capabilities

### New Capabilities

- None

### Modified Capabilities

- `quantization-runtime-metrics`: 明确 RQ-VAE eval payload、train/val/test loss 语义一致性，以及 RQ-VAE encoded/reconstruction 指标的命名与空间语义。

## Impact

- Affected code:
  - `src/quantization/rqvae/residual_quantization_vae.py`
  - `configs/model/rqvae_train.yaml`
- Affected tests:
  - RQ-VAE focused unit tests under `tests/quantization/`
- Affected specs:
  - `openspec/specs/quantization-runtime-metrics/spec.md`
- No dependency changes.
