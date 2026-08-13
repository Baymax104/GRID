## Why

Tail-SID analysis currently keeps a bespoke `LightningDataModule` under the quantization package, including single-batch dataset and collate mechanics. The data layer now has a stage-oriented DataModule hierarchy, so diagnosis experiments should use a horizontal data-layer path instead of keeping analysis-only loading code in the experiment package.

## What Changes

- Add a reusable diagnosis dataset and diagnosis DataModule under `src.data`.
- Represent diagnosis loading as an artifact-backed test-only dataset that can run preprocessing functions before yielding one full analysis batch.
- Move frequency-group assignment out of Tail-SID metrics into data preprocessing.
- Update Tail-SID analysis config and module imports to use the shared diagnosis data layer.
- Remove the Tail-SID-specific DataModule implementation after migration.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `datamodule-structure-alignment`: add a diagnosis DataModule variant alongside the file-backed DataModule.
- `tail-sid-resolution-diagnosis`: require Tail-SID analysis to load its test batch through the shared diagnosis DataModule/Dataset path.

## Impact

- Affected code: `src/data/datamodule`, `src/data/datasets.py`, `src/data/components/preprocessing.py`, `src/quantization/tail_sid_diagnosis`, and Tail-SID analysis configs/tests.
- Public config impact: `tail_sid_diagnosis` data target moves to `src.data.datamodule.DiagnosisDataModule`.
- Dependency impact: none.
