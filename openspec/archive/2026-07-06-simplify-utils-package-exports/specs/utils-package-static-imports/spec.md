## ADDED Requirements

### Requirement: utils package SHALL not depend on dynamic root exports
The `src.utils` package SHALL not use dynamic attribute-based export resolution as its primary public surface, and repository code SHALL be able to import utility symbols from concrete submodules directly.

#### Scenario: Root package no longer resolves exports dynamically
- **WHEN** a maintainer inspects `src/utils/__init__.py`
- **THEN** it MUST NOT use `__getattr__`-based export dispatch
- **THEN** it MUST NOT use `importlib.import_module` to lazily expose utility symbols

### Requirement: Repository callers SHALL import utility symbols from concrete submodules
Repository modules that currently depend on `src.utils` root re-exports SHALL import required symbols from their defining submodules instead of relying on package-level aggregation.

#### Scenario: Main entrypoint imports directly from submodules
- **WHEN** a maintainer inspects `src/main.py` or other repository callers previously importing symbols from `src.utils`
- **THEN** those modules MUST import the needed symbols from concrete modules such as `src.utils.pylogger`, `src.utils.utils`, or other defining submodules

### Requirement: utils internal modules SHALL avoid root-mediated sibling imports
Modules inside `src/utils` SHALL avoid importing sibling modules through `src.utils` root when a direct submodule import can express the dependency more clearly.

#### Scenario: Internal sibling import does not route through package root
- **WHEN** a maintainer inspects import statements inside `src/utils/*.py`
- **THEN** modules such as `logging_utils.py`, `instantiators.py`, `rich_utils.py`, and `utils.py` MUST prefer direct sibling submodule imports over `from src.utils import ...` patterns for internal dependencies
