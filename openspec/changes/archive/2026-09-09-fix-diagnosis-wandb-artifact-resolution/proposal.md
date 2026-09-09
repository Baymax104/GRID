## Why

Tail-SID diagnosis currently passes `semantic_id_path` and optional `embedding_path` directly to `DiagnosisDataset`, so short W&B run URIs reach the keyed bundle loader without explicit experiment `user/project` defaults and with the generic `model_output_path` field role. This makes the official diagnosis script fail before Artifact selection and also delays reference registration until after the lineage callback's setup window.

## What Changes

- Wire the official diagnosis data config's `${user}` and `${project}` into its test input assembly.
- Resolve Semantic ID and optional embedding references during `DiagnosisDataModule.setup`, using their actual field names so Artifact roles remain `semantic_id` and `semantic_embedding`.
- Pass resolved local bundle paths to `DiagnosisDataset`, preserving its keyed bundle parsing, item-key lookup, and local-path behavior.
- Ensure resolved diagnosis input references are registered before callback setup so the existing W&B lineage callback can record them.
- Add focused Hydra, DataModule, Artifact role, lineage timing, optional embedding, and local-path compatibility coverage.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tail-sid-resolution-diagnosis`: The official diagnosis DataModule must resolve Semantic ID and optional embedding inputs with experiment-provided W&B identity before constructing the diagnosis dataset.
- `keyed-prediction-bundle-artifact`: Diagnosis bundle references must preserve their semantic field roles while continuing to load the existing single-file keyed bundle protocol.
- `wandb-artifact-lineage`: Diagnosis input references must be available to the explicit lineage callback during Lightning setup.

## Impact

- Affected config: `configs/data/tail_sid_diagnosis.yaml`.
- Affected code: `src/data/datamodule/diagnosis.py`; `DiagnosisDataset`, the launcher, and W&B resolver APIs are expected to remain unchanged.
- Affected tests: diagnosis DataModule/config tests, W&B identity composition tests, and focused lineage/Artifact resolver coverage.
- External behavior: official diagnosis runs may continue using local paths, short `wandb://<run-id>` references, or fully qualified cross-project W&B URIs without requiring explicit `?role=` parameters for standard Semantic ID and embedding fields.
- Dependencies and bundle file formats remain unchanged.
