## Why

Analysis experiments now need the same execution pipeline as train and inference jobs: Hydra component assembly, DataModule loading, Trainer lifecycle, logger integration, callbacks, and artifact handling. Keeping Tail-SID diagnosis on a separate offline runner duplicates those responsibilities and makes W&B/artifact behavior diverge from the rest of the project.

## What Changes

- Route `run_mode: analysis` through the existing Lightning pipeline launcher.
- Execute analysis experiments through `Trainer.test(...)`, not train or predict.
- Add Tail-SID diagnosis DataModule and LightningModule components for test-only analysis.
- Keep Tail-SID machine-readable outputs and Markdown report, but emit them from the Lightning analysis lifecycle.
- Record numeric analysis summary metrics through the configured Lightning logger.
- Upload or expose diagnosis output files through callback/logger integration without binding quantization-domain code directly to W&B.

## Capabilities

### New Capabilities
- `lightning-analysis-test-pipeline`: Official analysis experiments run as test-only Lightning pipelines with DataModule, LightningModule, callbacks, trainer, and logger component config.

### Modified Capabilities
- `unified-main-entrypoint`: `run_mode: analysis` dispatches to the Lightning test pipeline instead of the offline runner path.
- `experiment-config-componentization`: analysis experiments compose `data`, `model`, `trainer`, `callbacks`, and `logger` component groups.
- `tail-sid-resolution-diagnosis`: Tail-SID diagnosis keeps the same metric/report outputs while running through Lightning test lifecycle.

## Impact

- Affected code: `src/main.py`, `src/utils/launcher.py`, `src/quantization/tail_sid_diagnosis/*`, `configs/experiment/tail_sid_diagnosis.yaml`, and new component config files.
- Affected tests: Tail-SID diagnosis tests, analysis dispatch tests, Hydra compose tests.
- No new external dependency is expected; W&B remains provided by the existing logger config.
