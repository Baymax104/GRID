## 1. 修复 ruff `exclude` 配置盲区

- [x] 1.1 修改 `pyproject.toml`：`[tool.ruff] exclude` 中 `"data"` → `"/data"`（锚定仓库根目录，解除对 `src/data/` 的误伤），其余 exclude 条目保持不变
- [x] 1.2 验证 `ruff check src/ --statistics` 现在能报出 `src/data/` 下的 UP 规则违规（即 `src/data/` 已重回 ruff 覆盖范围）

## 2. ruff 自动修复真类型标注

- [x] 2.1 跑 `uv run ruff check --fix --select UP006 --select UP007 --select UP035 --select UP045 src/`，自动修复全部真类型标注（`Dict/List/Tuple` → `dict/list/tuple`、`Optional[X]` → `X | None`、`Union[X,Y]` → `X | Y`）并清理对应 `from typing import` 废弃符号
- [x] 2.2 跑 `uv run ruff check --fix --select I src/` 收尾整理 import 排序（类型 import 清理后可能触发 `I001`）
- [x] 2.3 逐文件 `git diff` 复查，确认仅类型标注写法与 import 变动，无任何逻辑改动
- [x] 2.4 确认受影响的 9 个 `src/data/` 文件均已处理：`utils.py`、`datamodules/base.py`、`datamodules/sequence.py`、`datamodules/item.py`、`components/config_models.py`、`components/data_models.py`、`components/datasets.py`、`components/collate.py`、`components/preprocessing.py`

## 3. 手动修复 docstring 伪类型标注

- [x] 3.1 `src/utils/restart_job_utils.py`：Attributes 块 `List[Dict[str, Any]]` → `list[dict[str, Any]]`、`List[str]` ×2 → `list[str]`
- [x] 3.2 `src/utils/utils.py`：Returns 块 `Tuple[torch.nn.Module, str]` → `tuple[torch.nn.Module, str]`
- [x] 3.3 `src/utils/file_utils.py`：param 描述 `Optional[str]` → `str | None`
- [x] 3.4 `src/utils/decorators.py`：Args 块 `Optional[Type]` → `type | None`、`Optional[int]` ×6 → `int | None`、`Optional[bool]` → `bool | None`
- [x] 3.5 `src/utils/tensor_utils.py`：param 描述 `Optional[int]` → `int | None`
- [x] 3.6 `src/recommendation/base_recommender.py`：Returns 块 `Tuple[torch.Tensor, torch.Tensor]` → `tuple[torch.Tensor, torch.Tensor]`
- [x] 3.7 `src/recommendation/tiger_generation_model.py`：Parameters 块 `Optional[int]` / `Optional[torch.Tensor]` / `Optional[DynamicCache]` ×9 → 对应的 `X | None`
- [x] 3.8 `src/recommendation/decoder_module.py`：param 描述 `Optional[torch.nn.Parameter]` → `torch.nn.Parameter | None`
- [x] 3.9 `src/common/components/aggregation_strategy.py`：残缺行 `last_k Optional[int] = None` 顺手规范为 `last_k (int | None, optional):`
- [x] 3.10 `src/data/components/config_models.py`：numpy 风格 docstring 中 `Optional[dict]` / `Dict[str, callable]` / `Optional[Dict[...]]` / `Optional[List[str]]` / `Optional[int|bool]` ×14 → 对应内置泛型与管道写法
- [x] 3.11 `src/data/components/data_models.py`：numpy 风格 docstring 中 `Dict[...]` / `Union[...]` / `Optional[...]` ×8 → 对应内置泛型与管道写法
- [x] 3.12 `src/data/components/preprocessing.py`：param 描述 `Optional[list[str]]` → `list[str] | None`

## 4. 验证收尾

- [x] 4.1 `uv run ruff check --select UP006,UP007,UP035,UP045 src/` 确认全部清零（无任何 UP 类型规则违规）
- [x] 4.2 `rg` 复查 `src/` 下 docstring 残留：搜索 `Optional\[|Union\[|Dict\[|List\[|Tuple\[`，确认零残留（纯自然语言叙述此前已不在搜索范围内）
- [x] 4.3 `uv run python -m compileall src/ -q` 全部源码编译通过（exit 0），确认语法正确
- [x] 4.4 import smoke check：导入全部直接受影响模块，确认无循环导入或路径错误
- [x] 4.5 `uv run ruff check src/ --statistics` 确认整体 lint 状态：仅剩 5 个预先存在问题（F401×2、B008、F541、UP028），无因本次改动新增问题；`src/data/` 子目录 UP 规则维持 0 且已被 ruff 覆盖
