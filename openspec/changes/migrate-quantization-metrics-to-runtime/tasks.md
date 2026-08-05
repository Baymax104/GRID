## 1. Runtime Alignment

- [x] 1.1 Keep metric engine inputs required for configured quantization train metrics.
- [x] 1.2 Remove `log_every_n_steps` gating from quantization training metric payloads.

## 2. Quantization Models

- [x] 2.1 Migrate `ResidualKMeans` metric attributes, logging, and reset hooks to payload returns.
- [x] 2.2 Migrate `ResidualVectorQuantization` metric attributes, logging, and reset hooks to payload returns.
- [x] 2.3 Migrate `ResidualQuantizationVAE` metric attributes, logging, and reset hooks to payload returns.

## 3. Configs

- [x] 3.1 Add `model.metrics` to `rkmeans_train.yaml`.
- [x] 3.2 Add `model.metrics` to `rvq_train.yaml`.
- [x] 3.3 Add `model.metrics` to `rqvae_train.yaml`.

## 4. Tests and Validation

- [x] 4.1 Add tests proving quantization models no longer expose metric attributes.
- [x] 4.2 Add tests proving quantization configs declare repeat metrics.
- [x] 4.3 Run focused quantization and metric tests without full experiments.
- [x] 4.4 Run scoped ruff checks.
