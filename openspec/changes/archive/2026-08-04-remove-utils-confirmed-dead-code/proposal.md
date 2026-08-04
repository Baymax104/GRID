## Why

`src/utils/file.py` still contains helper functions that have no callers in Python code, Hydra `_target_` configuration, or tests. Keeping these functions preserves stale remote-file utility surface area and leaves the living `utils-dead-code-removal` spec with an outdated assertion that `SameFileError` must remain.

## What Changes

- Remove confirmed-dead `file.py` helpers:
  - `copy_to_remote`
  - `file_exists_local_or_remote`
  - `remove_file_extension`
- Remove imports that become orphaned after those functions are deleted.
- Update the `utils-dead-code-removal` contract so `file.py` dead-code cleanup includes these helpers and no longer requires `SameFileError`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `utils-dead-code-removal`: Extend the existing dead-code cleanup requirement to cover the newly confirmed unused `file.py` helpers and their orphan imports.

## Impact

- Affected code: `src/utils/file.py`.
- Affected specs: `openspec/specs/utils-dead-code-removal/spec.md`.
- No runtime API intentionally supported by current repo code is preserved for these helpers; all identified references are definitions, docstring examples, or stale OpenSpec text.
- No dependency changes.
