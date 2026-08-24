## 1. Regression Tests

- [x] 1.1 Add a focused CPU unit test proving RQ-VAE eval payload includes `loss`, `quantization_loss`, and `reconstruction_loss`.
- [x] 1.2 Add a focused CPU unit test proving RQ-VAE eval `loss` equals `quantization_loss_weight * quantization_loss + reconstruction_loss_weight * reconstruction_loss`.
- [x] 1.3 Add a focused CPU unit test proving encoded residual metrics use encoded embedding norm rather than raw input embedding norm.
- [x] 1.4 Add a focused config test proving `configs/model/rqvae_train.yaml` declares validation and test metrics for total, quantization, reconstruction, and space-explicit RQ-VAE stats.

## 2. RQ-VAE Metric Semantics

- [x] 2.1 Refactor RQ-VAE shared evaluation logic so train, validation, and test can compute total loss and decomposed loss fields consistently.
- [x] 2.2 Update RQ-VAE `eval_step` to compute decoder reconstruction when reconstruction loss is configured and layers are initialized.
- [x] 2.3 Update RQ-VAE output stats to emit encoded-space metrics with explicit names and encoded-space denominators.
- [x] 2.4 Add reconstruction-space metrics comparing decoded quantized embeddings against normalized input embeddings.
- [x] 2.5 Preserve existing prediction behavior and semantic ID output shape.

## 3. Metric Configuration

- [x] 3.1 Update RQ-VAE train metric config with space-explicit encoded and reconstruction metric names.
- [x] 3.2 Update RQ-VAE validation metric config to include `loss`, `quantization_loss`, `reconstruction_loss`, encoded-space stats, reconstruction-space stats, and semantic ID diversity stats.
- [x] 3.3 Update RQ-VAE test metric config to match validation metric semantics.

## 4. Verification

- [x] 4.1 Run the focused RQ-VAE and metric config unit tests with `uv run pytest`.
- [x] 4.2 Run Hydra compose or an equivalent lightweight config check for `experiment=rqvae_train`.
- [x] 4.3 Run `openspec validate fix-rqvae-metric-semantics --strict`.
- [x] 4.4 Inspect `git diff` to ensure the implementation changes are limited to RQ-VAE code, RQ-VAE config, tests, and the OpenSpec artifacts.
