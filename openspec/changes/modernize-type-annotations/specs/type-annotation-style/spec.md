## ADDED Requirements

### Requirement: 源码类型标注 SHALL 使用 PEP 585 与 PEP 604 内置语法
Python 源码中的类型标注 SHALL 使用内置泛型（`dict / list / tuple / set / frozenset` 等，PEP 585）与管道联合语法（`X | None`、`X | Y`，PEP 604），而非 `typing.Dict / List / Tuple / Optional / Union` 等已废弃的旧式别名。docstring 中以类型形式出现的描述 SHALL 与代码风格保持一致。

#### Scenario: 真类型标注使用内置泛型
- **WHEN** 维护者检查函数签名、变量标注或 dataclass 字段的类型标注
- **THEN** 标注 MUST NOT 使用 `typing.Dict / List / Tuple / Set / FrozenSet` 等已废弃别名
- **THEN** 标注 MUST 使用对应的内置泛型 `dict / list / tuple / set / frozenset`

#### Scenario: 可选与联合类型使用 PEP 604 管道语法
- **WHEN** 维护者检查可选类型或联合类型的标注
- **THEN** 标注 MUST NOT 使用 `typing.Optional[X]` 或 `typing.Union[X, Y]`
- **THEN** 标注 MUST 使用 `X | None` 或 `X | Y` 管道语法

#### Scenario: docstring 中的类型描述与代码风格一致
- **WHEN** 维护者检查 docstring 的 Args / Returns / Attributes 块中以类型形式出现的描述
- **THEN** 描述 MUST NOT 使用 `Optional[...] / Union[...] / Dict[...] / List[...] / Tuple[...]` 旧式写法
- **THEN** 描述 MUST 使用 `X | None / X | Y / dict[...] / list[...] / tuple[...]` 与代码一致的写法
- **THEN** 纯自然语言叙述（如 "Tuple of a tensor..."、"List of features..."）不在此约束范围内

#### Scenario: 废弃 typing 导入被清理
- **WHEN** 维护者检查 `from typing import ...` 语句
- **THEN** 该语句 MUST NOT 仅为导入 `Dict / List / Tuple / Optional / Union` 而存在
- **THEN** 仅剩这些废弃符号的 import 语句 MUST 被整行删除；与其他仍需保留符号（如 `Any / TypeVar / Literal`）共存的语句 MUST 仅移除废弃符号

### Requirement: ruff `exclude` 配置 SHALL 精确锚定顶层目录
`pyproject.toml` 的 `[tool.ruff] exclude` 配置 SHALL 精确锚定需排除的顶层目录，不得以裸名称形式误伤 `src/` 下的同名子目录，确保类型标注类规则（及全部 lint 规则）覆盖所有源码目录。

#### Scenario: 顶层 data 目录排除条目锚定仓库根
- **WHEN** 维护者检查 `[tool.ruff] exclude` 中针对顶层 `data/` 数据文件夹的排除条目
- **THEN** 该条目 MUST 写作 `"/data"`（以 `/` 锚定仓库根目录）
- **THEN** 该条目 MUST NOT 写作裸 `"data"`（会按 ruff 路径段匹配规则误伤 `src/data/`）

#### Scenario: src/data 目录受 ruff 覆盖
- **WHEN** 维护者对 `src/` 运行 `ruff check`
- **THEN** `src/data/` 子目录 MUST 被 ruff 扫描覆盖
- **THEN** `src/data/` 下的 UP 规则违规（如 `UP006 / UP007 / UP035 / UP045`）MUST 能被正常报出
