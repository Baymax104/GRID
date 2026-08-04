## MODIFIED Requirements

### Requirement: Repository callers SHALL import utility symbols from concrete submodules
Repository modules that currently depend on `src.utils` root re-exports SHALL import required symbols from their defining submodules instead of relying on package-level aggregation.

#### Scenario: Main entrypoint imports directly from submodules
- **WHEN** a maintainer inspects `src/main.py` or other repository callers previously importing symbols from `src.utils`
- **THEN** those modules MUST import the needed symbols from concrete modules such as `src.utils.pylogger`, `src.utils.extra`, `src.utils.model`, or other defining submodules
- **THEN** those modules MUST NOT import from removed modules such as `src.utils.startup` or `src.utils.model_utils`

### Requirement: utils internal modules SHALL avoid root-mediated sibling imports
Modules inside `src/utils` SHALL avoid importing sibling modules through `src.utils` root when a direct submodule import can express the dependency more clearly.

#### Scenario: Internal sibling import does not route through package root
- **WHEN** a maintainer inspects import statements inside `src/utils/*.py`
- **THEN** modules such as `logging.py`, `rich.py`, `extra.py`, and `model.py` MUST prefer direct sibling submodule imports over `from src.utils import ...` patterns for internal dependencies
- **THEN** utils modules MUST NOT import removed sibling module paths with `_utils` suffixes
