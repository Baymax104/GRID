## Why

`FileDataModule` currently owns both Lightning stage lifecycle behavior and file-backed dataset assembly. Recent stage handling fixes made the lifecycle responsibility explicit, and extracting it now gives the data layer a stable base for future DataModule variants without changing official file-backed pipelines.

## What Changes

- Add a shared stage-oriented DataModule base that owns stage configuration storage, setup-stage resolution, and dataloader hook dispatch.
- Change `FileDataModule` to inherit from that stage base while retaining file discovery, worker file assignment, dataset construction, collate selection, and `DataloaderWithIterationRetry` assembly.
- Update official Hydra datamodule targets to `src.data.datamodule.FileDataModule`.
- Do not connect analysis DataModules in this change; this only prepares the inheritance structure.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `datamodule-structure-alignment`: clarify that shared stage lifecycle mechanics are centralized separately from file-backed loading mechanics.

## Impact

- Affected code: `src/data/datamodule/file.py`, new `src/data/datamodule/stage.py`, and focused data module tests.
- Public config impact: official file-backed datamodule targets move from `src.data.data_module.BaseDataModule` to `src.data.datamodule.FileDataModule`.
- Dependency impact: none.
