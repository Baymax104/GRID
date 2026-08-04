## Context

The utils package already follows a direct-submodule import policy: repository callers import concrete modules rather than symbols from the package root. That means a file rename requires mechanical updates across Python imports and OpenSpec references, but does not require an API compatibility layer.

Current rename targets:

- `cli_utils.py` -> `cli.py`
- `distributed_utils.py` -> `distributed.py`
- `file_utils.py` -> `file.py`
- `launcher_utils.py` -> `launcher.py`
- `logging_utils.py` -> `logging.py`
- `model_utils.py` -> `model.py`
- `rich_utils.py` -> `rich.py`
- `startup.py` -> `extra.py`

## Goals / Non-Goals

**Goals:**

- Remove `_utils` suffixes from utility module filenames.
- Rename startup extras module to `extra.py`.
- Update all active Python imports and documentation references.
- Keep `hydra_resolvers.py`, `decorators.py`, `pylogger.py`, and `__init__.py` unchanged.
- Avoid compatibility stubs so residual scans can detect stale imports.

**Non-Goals:**

- Do not move functions between modules.
- Do not rename function or class symbols such as `extras`, `pipeline_launcher`, or `RankedLogger`.
- Do not alter runtime behavior.
- Do not update archived OpenSpec history.

## Decisions

1. Remove old module files instead of keeping forwarding stubs
   - Rationale: this is an internal repository import-path cleanup, and stubs would preserve stale paths.
   - Alternative considered: leave deprecated wrappers. Rejected because the user asked to delete suffixes, not maintain aliases.

2. Use fully qualified imports for potentially ambiguous names
   - Rationale: `src.utils.logging` and `src.utils.rich` are clear when imported by full package path; bare `import logging` should continue to refer to stdlib logging.
   - Alternative considered: avoid `logging.py` because of stdlib name overlap. Rejected because the requested suffix-removal rule naturally yields `logging.py`, and current code can use qualified imports.

3. Keep `hydra_resolvers.py` unchanged
   - Rationale: it does not end with `_utils`, and renaming resolver registration modules is outside the requested scope.
   - Alternative considered: rename it to `hydra.py`. Rejected as unrelated scope expansion.

## Risks / Trade-offs

- [Risk] Missed import path in a config/spec/test breaks runtime import -> Mitigation: scan `src`, `configs`, `tests`, active OpenSpec specs/changes, and docs for old module names.
- [Risk] Local `src.utils.logging` could be confused with stdlib `logging` -> Mitigation: use full `src.utils.logging` import paths and avoid bare imports for the local module.
