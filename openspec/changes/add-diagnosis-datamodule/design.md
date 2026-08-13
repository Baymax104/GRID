## Context

Tail-SID diagnosis needs a test-only full-batch input assembled from:

- Semantic ID keyed prediction bundle
- Optional item embedding keyed prediction bundle
- Training sequence rows under `data_dir/training`
- Frequency-derived item groups

The current implementation builds this batch in `src.quantization.tail_sid_diagnosis.datamodule`. That duplicates DataModule lifecycle mechanics and keeps data preprocessing under the analysis package.

## Design

Add `DiagnosisDataModule` in `src/data/datamodule/diagnosis.py`.

Responsibilities:

- Inherit `StageDataModule`
- Accept `test_dataloader_config`
- Set up only `TrainerFn.TESTING`
- Instantiate the configured `DiagnosisDataset`
- Return a standard `torch.utils.data.DataLoader`

Add `DiagnosisDataset` in `src/data/datasets.py`.

Responsibilities:

- Implement `__len__ == 1`
- Load semantic ID views from a keyed prediction bundle
- Load optional item embeddings by item id
- Iterate training rows through `TFRecordReader`
- Compute per-item training frequencies
- Build a mutable diagnosis batch row
- Apply configured preprocessing functions
- Return the final diagnosis batch

The artifact loading and frequency computation helpers SHALL be private methods on `DiagnosisDataset`, not module-level public functions.

Add diagnosis data models in `src/data/components/data_models.py`:

- `SIDViews`
- `DiagnosisBatch`

Add preprocessing in `src/data/components/preprocessing.py`:

- `assign_frequency_groups`

The preprocessing function receives a diagnosis batch, computes Head/Mid/Tail/Tail-Cold groups from frequencies, and returns the updated batch.

## Dependency Direction

`src.data` must not import `src.quantization.tail_sid_diagnosis.metrics`.

Tail-SID metrics and module code may import `DiagnosisBatch` and `SIDViews` from the data layer. This preserves the direction:

```text
data -> generic loading and preprocessing
tail_sid_diagnosis -> metric computation and reporting
```

## Non-Goals

- Do not change Tail-SID metric formulas or output schemas.
- Do not add a generic multi-batch analysis framework.
- Do not migrate report writing in this change.
