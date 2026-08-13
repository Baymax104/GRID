## Context

The Lightning migration makes `cfg.analysis.runner` obsolete for official experiments. Keeping `src/common/analysis/run_analysis_runner`, `configs/analysis/tail_sid_diagnosis.yaml`, and Tail-SID `runner.py` would preserve a second analysis lifecycle with separate logger and artifact semantics.

## Goals / Non-Goals

**Goals:**
- Remove the offline analysis runner contract from official runtime behavior.
- Delete stale analysis config and runner code after the Lightning path is in place.
- Remove temporary non-Lightning logger lifecycle helpers that exist only for offline analysis.
- Update tests and specs so the only official analysis path is Lightning `test`.

**Non-Goals:**
- Do not remove `run_mode: analysis`.
- Do not remove Tail-SID reporting or metric APIs.
- Do not change train or inference lifecycle behavior.

## Decisions

1. Delete the runner contract instead of keeping compatibility re-exports

   The project already treats official entrypoints as explicitly migrated rather than maintaining parallel APIs. Keeping compatibility wrappers would make it unclear which lifecycle owns logging and artifacts.

2. Preserve domain helpers but remove orchestration wrappers

   `data.py`, `metrics.py`, and `reporting.py` remain valid domain modules. `runner.py` is orchestration and should be replaced by DataModule/LightningModule/callback components.

## Risks / Trade-offs

- Existing local scripts or ad hoc imports of `run_analysis_runner` will break -> official scripts should use `src.main experiment=tail_sid_diagnosis`, and tests will scan for old references.
- Cleanup depends on the migration being functional -> execute cleanup only after Lightning tests pass.

## Migration Plan

1. Verify Tail-SID diagnosis runs through Lightning test.
2. Delete `src/common/analysis/` runner exports and tests.
3. Delete `configs/analysis/tail_sid_diagnosis.yaml`.
4. Delete Tail-SID `runner.py`.
5. Remove temporary logger helpers from `src/utils/logging.py` if no longer used.
6. Run residual reference scans and focused tests.
