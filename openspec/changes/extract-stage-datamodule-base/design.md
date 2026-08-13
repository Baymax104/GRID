## Context

`FileDataModule` is the official DataModule implementation for file-backed train, validation, test, and predict pipelines. It currently mixes two responsibilities:

- Lightning stage lifecycle mechanics: storing stage configs, resolving which stages to prepare for a `setup(stage)` call, and dataloader hook dispatch.
- File-backed loading mechanics: listing files, assigning files to ranks, constructing iterable datasets, choosing collate functions, and returning `DataloaderWithIterationRetry`.

The first responsibility can be shared by future DataModule variants. The second should remain local to `FileDataModule`.

## Design

Introduce `StageDataModule` in `src/data/datamodule/stage.py`.

`StageDataModule` owns:

- `stage_to_config`
- `_resolve_setup_stages(stage)`
- `setup(stage)`
- `get_stage_config(stage)`
- `get_dataloader(stage)`
- `train_dataloader`, `val_dataloader`, `test_dataloader`, and `predict_dataloader`

`StageDataModule` requires subclasses to implement:

- `setup_stage(stage)`
- `build_dataloader(stage)`

`FileDataModule` then owns only file-backed behavior:

- `stage_to_file_map`
- idempotent file-map preparation
- file-map readiness checks before dataloader construction
- file suffix resolution
- file listing and limit application
- worker/rank file assignment
- dataset construction
- persistent worker resolution
- collate function resolution
- `DataloaderWithIterationRetry` construction

## Stage Semantics

The base lifecycle maps Lightning setup stages as follows:

- `None`: all configured stages
- `fit`: `FITTING` and `VALIDATING`
- `validate`: `VALIDATING`
- `test`: `TESTING`
- `predict`: `PREDICTING`

The `fit` mapping includes validation because `Trainer.fit()` calls `setup("fit")`, while the fit lifecycle may still request validation dataloaders.

## Non-Goals

- Do not add an analysis or artifact-backed DataModule in this change.
- Do not change official Hydra datamodule targets.
- Do not change dataset, reader, collate, or `DataloaderWithIterationRetry` behavior.
