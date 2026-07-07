## Context

项目 `pyproject.toml` 已设定 `target-version = "py311"`，且 `[tool.ruff.lint] select` 已包含 `UP`（pyupgrade）规则。理论上 `UP006`（PEP 585）、`UP007/UP045`（PEP 604）、`UP035`（废弃 typing 导入）应当强制把 `Dict/List/Tuple/Optional/Union` 现代化。但实际存在两个阻断因素：

1. **lint 盲区**：`[tool.ruff] exclude = ["data"]` 按 ruff 的路径段匹配规则，误伤了 `src/data/` 子目录。而项目里真正集中的旧式类型标注恰好都在 `src/data/` 下（config_models、data_models、collate、preprocessing 等）。整目录 `ruff check src/` 永远跳过该目录，导致这些旧式标注从未被捕获。单文件显式传入虽能绕过 exclude，但日常 lint 不这么做。
2. **docstring 伪标注**：约 26 处 `Optional[...] / Union[...] / Dict[...] / List[...] / Tuple[...]` 出现在文档字符串的 `Args/Returns/Attributes` 块里，属于文本而非真类型标注，ruff 不覆盖，需手动处理。

此外，`src/data/components/preprocessing.py` 已部分现代化（用 `list[str]` 但仍保留 `Optional[...]`），状态不一致。全项目无 `from __future__ import annotations`，标注在运行时求值，但 grep 确认无 `get_type_hints / __annotations__ / == Optional` 等运行时类型反射，py311 原生支持 `X | None`（`types.UnionType`），改造对运行时语义无影响。

## Goals / Non-Goals

**Goals:**
- 将全部源码真类型标注迁移到 PEP 585（`dict/list/tuple`）与 PEP 604（`X | None` / `X | Y`）
- 同步更新 docstring 中的伪类型标注，保持代码与文档风格一致
- 修复 ruff `exclude` 配置，解除对 `src/data/` 的误伤，恢复整目录 lint 覆盖
- 验证运行时无回归（语法、导入、lint 复查）

**Non-Goals:**
- 不引入 `from __future__ import annotations`（保持运行时求值语义，最小改动）
- 不重构任何类型逻辑或字段结构，仅改标注写法
- 不处理 `UP028`（yield-in-for-loop）等非类型类规则
- 不排查其他潜在 lint 盲区（仅修 `exclude` 这一处对 `src/data/` 的误伤）

## Decisions

### 决策 1：ruff `--fix` 自动修真标注，手动修 docstring

真类型标注位于明确的语法位置（函数签名、变量标注、dataclass 字段），ruff `--fix` 可靠且能同步清理 `from typing import` 中的废弃符号。docstring 里的伪标注格式多样，且包含 `Optional[Type]`（`Type` 指 `typing.Type`）等需逐处判断的写法，ruff 不覆盖，手动 edit 更可控。

**规则选择**：用 `--select UP006 --select UP007 --select UP035 --select UP045`（多次 `--select`）精准限定这 4 类规则。
- 备选方案 A：`--select UP`（整个 pyupgrade 类别）——会额外报出 `UP028` 等非类型规则噪音，且与本任务范围不符。
- 备选方案 B：依赖配置自带的 `select = ["...", "UP", "..."]` 跑 `ruff check --fix src/`——会同时修复 `I001`（import 排序）、`F401`（未使用导入）等，改动范围超出本任务，diff 难审。
- 选定方案的优势：精准只动类型标注相关规则，diff 纯净，易于复查。

### 决策 2：先修 `exclude`，再整目录 fix

顺序：先改 `pyproject.toml` 的 `exclude`（`"data"` → `"/data"`），让 `src/data/` 重回 ruff 覆盖范围；再跑 `ruff check --fix --select UP006 --select UP007 --select UP035 --select UP045 src/` 整目录修复。这样无需对 `src/data/` 单独显式传路径，流程统一。
- 备选方案：先对 `src/data/` 显式 fix 再改 exclude——多一步且顺序割裂，无优势。

### 决策 3：`exclude` 用 `/data` 锚定根目录

ruff 的 `exclude` 路径以 `/` 开头时锚定项目根。`"/data"` 只匹配根目录的 `data/`（数据文件夹，本就该排除），不再匹配 `src/data/`。其余 exclude 条目（`.git`、`.idea`、`.venv`、`logs`、`openspec`、`outputs`、`pretrained_models`）名称特殊，不会与 `src/` 下子目录冲突，保持不动以最小化改动。
- 备选方案：给所有条目加 `/` 前缀——更"严格"但属于无关改动，增加 diff 噪音，不采纳。

### 决策 4：docstring 中 `Optional[Type]` 的处理

`src/utils/decorators.py:96` 的 docstring 写的是 `exception_to_check (Optional[Type])`，其中 `Type` 指 `typing.Type`。统一改为 `type | None`（与 PEP 585 的 `type` 内置一致）。该文件未导入 `Type`，纯文档文本，改动不影响代码。

### 决策 5：顺带修正 `aggregation_strategy.py` 残缺 docstring 行

`src/common/components/aggregation_strategy.py:44` 的 `last_k Optional[int] = None` 缺冒号、格式本就残缺（代码本身 line 39 已是 `last_k: int | None = None`）。借本次类型现代化顺手规范为 Google 风格的 `last_k (int | None, optional):`，避免留下不一致。

## Risks / Trade-offs

- **[风险] ruff `--fix` 误改** → 缓解：`--select` 精准限定 4 规则；fix 后逐文件 diff 复查；`py_compile` 验证语法。
- **[风险] docstring 手改遗漏或格式错误** → 缓解：改后 `rg` 复查 `Optional\[|Union\[|Dict\[|List\[|Tuple\[` 在 `src/` 下的残留；仅放过纯自然语言描述（如 `Tuple of a tensor...`）。
- **[风险] `exclude` 改动意外影响其他目录扫描** → 缓解：仅改 `"data"` 一条；改后 `ruff check src/ --statistics` 确认 `src/data/` 下的 UP 规则被报出（即已覆盖）。
- **[权衡] 不引入 `__future__ annotations`** → 标注保持运行时求值，但 py311 原生支持 `X | None`，且无运行时类型反射依赖，功能无影响；保持最小改动面。
- **[权衡] `Optional[X]` → `X | None` 改变了运行时类型对象**（`typing.Optional[X]` vs `types.UnionType`）→ 已确认无代码依赖二者相等性，属于可接受的安全改动。

## Migration Plan

1. 改 `pyproject.toml`：`exclude` 中 `"data"` → `"/data"`
2. 跑 `ruff check --fix --select UP006 --select UP007 --select UP035 --select UP045 src/` 修复全部真标注与对应 import
3. 手动 edit 8 个文件约 26 处 docstring 伪标注
4. 验证：`ruff check --select UP src/` 确认 UP006/UP007/UP035/UP045 清零；`rg` 复查 docstring 残留；`uv run python -m py_compile` 受影响文件
5. `ruff check --select I --fix src/` 收尾整理 import 排序（类型 import 清理后可能触发 I001）
6. 单一 commit 提交全部改动

## Open Questions

无。所有技术决策已在前序探索中澄清（运行时安全性、规则选择、docstring 范围、exclude 修复方式均确认）。
