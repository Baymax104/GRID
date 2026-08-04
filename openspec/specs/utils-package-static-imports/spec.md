## Purpose

规范 `src/utils/` 包中工具函数的导入方式，要求存储库模块从具体的子模块导入工具符号，而非通过包级动态导出。同时要求 utils 内部模块使用直接的兄弟子模块导入，避免通过包根路径间接引用。
## Requirements
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

