## Why

项目已 `target-version = "py311"`，但源码中仍残留大量 `typing.Dict / List / Tuple / Optional / Union` 旧式标注，未迁移到 PEP 585（`dict / list / tuple`）与 PEP 604（`X | None` / `X | Y`）内置语法。更严重的是，`pyproject.toml` 的 `ruff exclude = ["data"]` 本意排除根目录数据文件夹，却因 ruff 的 exclude 按路径段匹配而**误伤 `src/data/` 整个子目录**，使其成为 lint 盲区——`src/data/` 下集中的旧式标注从未被任何 lint 检查捕获。现在统一现代化标注语法，并修复该 exclude 盲区，让 ruff 重新覆盖全部源码。

## What Changes

- 将所有 `Dict[...] / List[...] / Tuple[...]` 类型标注改为 `dict[...] / list[...] / tuple[...]`（PEP 585）
- 将所有 `Optional[X]` 改为 `X | None`，`Union[X, Y, ...]` 改为 `X | Y | ...`（PEP 604）
- 清理 `from typing import ...` 中因此变为未使用的 `Dict / List / Tuple / Optional / Union` 导入；对仅剩这些符号的 import 语句整行删除
- 同步更新 docstring 中以 `Optional[...] / Union[...] / Dict[...] / List[...] / Tuple[...]` 形式出现的伪类型标注，保持代码与文档风格一致
- **修复 ruff `exclude` 配置**：`"data"` → `"/data"`（锚定仓库根目录），解除对 `src/data/` 的误伤；其余 exclude 条目因不会匹配 `src/` 下子目录，保持不变
- 顺带修正 `src/common/components/aggregation_strategy.py` docstring 中格式本就残缺的一行（`last_k Optional[int] = None` → 规范的 `(int | None, optional)`）

## Capabilities

### New Capabilities

- `type-annotation-style`: 规定源码类型标注 SHALL 使用 PEP 585 内置泛型（`dict/list/tuple`）与 PEP 604 管道联合语法（`X | None` / `X | Y`），并要求 ruff `exclude` 配置精确锚定顶层目录、不得误伤 `src/` 下同名子目录，使类型标注规则能覆盖全部源码

### Modified Capabilities

无。现有 specs（`data-model-role-separation`、`python-component-entrypoint-normalization`、`utils-package-static-imports`、`data-precomputed-lookup`、`keyed-prediction-bundle-artifact`）均不涉及类型标注风格约束，本次不修改任何现有 spec 的 requirement。

## Impact

- **真类型标注（`src/data/` 下，ruff `--fix` 自动处理）**：
  - `src/data/utils.py`、`src/data/datamodules/base.py`、`src/data/datamodules/sequence.py`、`src/data/datamodules/item.py`
  - `src/data/components/config_models.py`、`src/data/components/data_models.py`、`src/data/components/datasets.py`、`src/data/components/collate.py`、`src/data/components/preprocessing.py`
- **docstring 伪类型标注（手动处理，约 26 处）**：
  - `src/utils/restart_job_utils.py`、`src/utils/utils.py`、`src/utils/file_utils.py`、`src/utils/decorators.py`、`src/utils/tensor_utils.py`
  - `src/recommendation/base_recommender.py`、`src/recommendation/tiger_generation_model.py`、`src/recommendation/decoder_module.py`
  - `src/common/components/aggregation_strategy.py`
- **配置**：`pyproject.toml`（`[tool.ruff] exclude`）
- **运行时行为**：不变。全项目无 `get_type_hints` / `__annotations__` / `== Optional` 等运行时类型反射；py311 原生支持 `X | None` 求值为 `types.UnionType`，`@dataclass` 字段能正确处理。改造对运行时语义无影响。
- **非 BREAKING**：仅现代化标注写法与 docstring 文本，不改变任何公共 API 签名、字段或行为。
