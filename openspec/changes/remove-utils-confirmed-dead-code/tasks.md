## 1. Code Cleanup

- [x] 1.1 Remove `copy_to_remote`, `file_exists_local_or_remote`, and `remove_file_extension` from `src/utils/file_utils.py`.
- [x] 1.2 Remove imports that become unused after the function deletions, including `SameFileError` and `_PATH`.

## 2. Verification

- [x] 2.1 Search `src`, `configs`, and `tests` for the removed function names to confirm no active references remain.
- [x] 2.2 Run `uv run ruff check src\utils\file_utils.py`.
- [x] 2.3 Run OpenSpec status or validation for `remove-utils-confirmed-dead-code`.
