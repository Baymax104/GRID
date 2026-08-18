## Why

Once official analysis experiments run through the Lightning test pipeline, the old offline runner contract becomes a parallel lifecycle with duplicated logger and artifact behavior. Removing it keeps runtime semantics clear: official train, inference, and analysis jobs all enter through `src.main` and use componentized pipeline assembly.

## What Changes

- **BREAKING**: remove the official `cfg.analysis.runner` Hydra contract for analysis experiments.
- Remove `src/common/analysis/run_analysis_runner` and its tests after Tail-SID diagnosis no longer uses it.
- Remove `configs/analysis/tail_sid_diagnosis.yaml`.
- Remove Tail-SID `runner.py` once its computation/reporting behavior has moved into Lightning module/callback components.
- Remove the temporary non-Lightning logger lifecycle added for offline analysis.
- Keep `run_mode: analysis` as an official mode, but define it as Lightning `test` execution.

## Capabilities

### New Capabilities

### Modified Capabilities
- `analysis-runner-contract`: retire the offline runner contract and replace it with the Lightning analysis pipeline contract.
- `unified-main-entrypoint`: remove the analysis runner dispatch path.
- `experiment-config-componentization`: remove `/analysis@analysis` as the official analysis assembly node.
- `tail-sid-resolution-diagnosis`: remove the Hydra-instantiable Tail-SID analysis runner requirement.

## Impact

- Affected code: `src/common/analysis/`, `src/main.py`, `src/utils/logging.py`, `src/quantization/tail_sid_diagnosis/runner.py`.
- Affected config: `configs/analysis/`, `configs/experiment/tail_sid_diagnosis.yaml`.
- Affected tests: analysis runner tests become obsolete; Tail-SID diagnosis coverage moves to DataModule/LightningModule/callback tests.
