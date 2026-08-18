## 1. Lightning Analysis Components

- [x] 1.1 Add a Tail-SID diagnosis DataModule that loads SID views, optional embeddings, and training frequencies for the test lifecycle.
- [x] 1.2 Add a Tail-SID diagnosis LightningModule that computes `DiagnosisResult` during `test_step` and logs numeric summary metrics.
- [x] 1.3 Add a Tail-SID diagnosis report callback that writes the existing JSON, CSV, and Markdown outputs after test computation.

## 2. Pipeline Wiring

- [x] 2.1 Change `run_mode: analysis` dispatch to use `pipeline_launcher` and `Trainer.test(...)`.
- [x] 2.2 Add Tail-SID `data`, `model`, `callbacks`, `trainer`, and `logger` component configs.
- [x] 2.3 Update `configs/experiment/tail_sid_diagnosis.yaml` to compose the Lightning analysis components.

## 3. Verification

- [x] 3.1 Add or update unit tests for Tail-SID DataModule, LightningModule, callback, and Hydra composition.
- [x] 3.2 Run focused pytest for analysis and Tail-SID diagnosis.
- [x] 3.3 Run scoped Ruff on touched Python files.
