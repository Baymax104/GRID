## Why

`src/utils/` still contains several modules with redundant `_utils` suffixes, while `startup.py` hosts the `extras` startup hook under a less direct name. Renaming these modules makes utility import paths shorter and aligns module names with their actual responsibility.

## What Changes

- Rename `_utils` suffixed modules in `src/utils/` to concise responsibility names:
  - `cli_utils.py` -> `cli.py`
  - `distributed_utils.py` -> `distributed.py`
  - `file_utils.py` -> `file.py`
  - `launcher_utils.py` -> `launcher.py`
  - `logging_utils.py` -> `logging.py`
  - `model_utils.py` -> `model.py`
  - `rich_utils.py` -> `rich.py`
- Rename `startup.py` to `extra.py`.
- Update Python imports, repository documentation, and OpenSpec requirements to use the new module names.
- **BREAKING**: Old `src.utils.*_utils` and `src.utils.startup` import paths are removed; no compatibility stubs are kept.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `utils-single-responsibility-modules`: Update the canonical module names for startup extras and model helpers.
- `utils-package-static-imports`: Update required concrete import examples and internal sibling-import examples.
- `utils-dead-code-removal`: Update file-specific dead-code cleanup requirements to the new `file.py` and `logging.py` module names.

## Impact

- Affected code:
  - `src/utils/*.py`
  - `src/main.py`
  - callers under `src/data`, `src/inference`, `src/quantization`, `src/recommendation`, and `src/common`
  - focused quantization tests that import distributed helpers
- Affected docs/specs:
  - `AGENTS.md`
  - utils-related OpenSpec living specs
- No dependency changes.
