## Context

The existing `utils-dead-code-removal` specification already establishes that unused functions in `src/utils/` should be deleted when they have no Python callers, YAML `_target_` references, or OmegaConf resolver references. A fresh scan found three additional `src/utils/file_utils.py` helpers with no real callers: `copy_to_remote`, `file_exists_local_or_remote`, and `remove_file_extension`.

The living spec still says `SameFileError` must remain because it is used by `copy_to_remote`. That statement is now stale because `copy_to_remote` itself is confirmed unused.

## Goals / Non-Goals

**Goals:**

- Remove only the confirmed-dead `file_utils` helpers.
- Clean imports made orphaned by that removal.
- Update the spec contract so future dead-code checks do not preserve `SameFileError` for a deleted function.
- Validate with focused search and Ruff.

**Non-Goals:**

- Do not remove `retry`, `timeout`, or timeout-related exception classes in this change. They need a separate API decision because `timeout` is still reachable through optional `retry` parameters even though those branches are not currently enabled.
- Do not reorganize `src/utils/` module boundaries.
- Do not change runtime behavior for active data loading, tensor loading, checkpoint lookup, logging, or prediction writing paths.

## Decisions

1. Treat config references as first-class usage.
   - Rationale: Hydra `_target_` strings are runtime entrypoints even when static Python import scans cannot see them.
   - Alternative considered: Python-only reference search. Rejected because it would incorrectly mark configured loaders such as `load_model_output` as dead.

2. Limit implementation to functions with zero real callers.
   - Rationale: `copy_to_remote`, `file_exists_local_or_remote`, and `remove_file_extension` have no callers outside definitions, docstring examples, or stale OpenSpec text.
   - Alternative considered: Remove the unused `retry` timeout branch at the same time. Rejected because that is an API-surface simplification, not the same class of confirmed dead code.

3. Update the living spec through a delta.
   - Rationale: The current spec explicitly preserves `SameFileError`; implementation without a spec correction would leave the contract inconsistent.
   - Alternative considered: Leave the spec unchanged and only update code. Rejected because future validation would preserve an import that should be removed with `copy_to_remote`.

## Risks / Trade-offs

- `copy_to_remote` may have been intended for future remote artifact handling -> Mitigation: no current repo code calls it; reintroduce it later only with a concrete caller and tests.
- OpenSpec archival could accidentally drop existing scenarios if the delta is partial -> Mitigation: use `MODIFIED Requirements` with the full updated requirement blocks.
- Search-based dead-code detection can miss dynamic attribute access -> Mitigation: restrict deletion to module-level functions with no string `_target_` use and no package-level re-export.
