## 1. Module Renames

- [x] 1.1 Rename `src/utils/cli_utils.py` to `src/utils/cli.py`.
- [x] 1.2 Rename `src/utils/distributed_utils.py` to `src/utils/distributed.py`.
- [x] 1.3 Rename `src/utils/file_utils.py` to `src/utils/file.py`.
- [x] 1.4 Rename `src/utils/launcher_utils.py` to `src/utils/launcher.py`.
- [x] 1.5 Rename `src/utils/logging_utils.py` to `src/utils/logging.py`.
- [x] 1.6 Rename `src/utils/model_utils.py` to `src/utils/model.py`.
- [x] 1.7 Rename `src/utils/rich_utils.py` to `src/utils/rich.py`.
- [x] 1.8 Rename `src/utils/startup.py` to `src/utils/extra.py`.

## 2. Import Migration

- [x] 2.1 Update Python imports in `src/main.py`.
- [x] 2.2 Update Python imports under `src/data` and `src/inference`.
- [x] 2.3 Update Python imports under `src/quantization`, `src/recommendation`, and `src/common`.
- [x] 2.4 Update Python imports inside `src/utils` modules.
- [x] 2.5 Update focused tests importing renamed utils modules.

## 3. Documentation and Specs

- [x] 3.1 Update `AGENTS.md` references to renamed utils modules.
- [x] 3.2 Update living utils OpenSpec specs to use new module names.

## 4. Verification

- [x] 4.1 Run residual scans for old utils module paths and removed filenames outside archive history.
- [x] 4.2 Run `openspec validate rename-utils-modules --strict`.
- [x] 4.3 Run `openspec validate remove-utils-confirmed-dead-code --strict`.
- [x] 4.4 Run focused `uv run ruff check` for touched code and tests.
- [x] 4.5 Run focused tests covering renamed utils imports.
- [x] 4.6 Run Hydra compose smoke for main entrypoint imports.
