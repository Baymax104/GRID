## 1. 新建模块并迁移函数

- [x] 1.1 创建 `src/utils/startup.py`，从 `utils.py` 迁移 `extras()` 和 `print_warnings_for_missing_configs()`，保留对 `src.utils.pylogger.RankedLogger` 和 `src.utils.rich_utils.print_config_tree` 的导入
- [x] 1.2 创建 `src/utils/model_utils.py`，从 `utils.py` 迁移 `delete_module` / `find_module_shape` / `reset_parameters` / `get_parent_module_and_attr`，从 `tensor_utils.py` 迁移 `create_last_k_mask`
- [x] 1.3 创建 `src/data/components/tokenization.py`，从 `utils.py` 迁移 `load_tokenize()`，保留对 `functools.partial` 的依赖

## 2. 修复已知问题

- [x] 2.1 将 `src/utils/custom_hydra_resolvers.py` 重命名为 `src/utils/hydra_resolvers.py`（mv）
- [x] 2.2 在 `src/main.py` 添加 `import src.utils.hydra_resolvers  # noqa: F401`（恢复 `now_tz` resolver 注册），置于 `rootutils.setup_root` 之后、其他 utils import 之前
- [x] 2.3 修复 `src/utils/decorators.py` 的 `timeout` 装饰器：在模块顶层检测 `hasattr(signal, "SIGALRM")`，不可用时 `raise NotImplementedError("timeout decorator requires Unix signal.SIGALRM")`，而非直接引用 `signal.SIGALRM` 导致 `AttributeError`

## 3. 更新外部消费者导入

- [x] 3.1 `src/main.py`：`from src.utils.utils import extras` → `from src.utils.startup import extras`
- [x] 3.2 `src/recommendation/encoder_module.py`：`from src.utils.utils import delete_module, find_module_shape, reset_parameters` → `from src.utils.model_utils import ...`
- [x] 3.3 `src/recommendation/decoder_module.py`：`from src.utils.utils import delete_module, reset_parameters` → `from src.utils.model_utils import ...`
- [x] 3.4 `src/recommendation/tiger_generation_model.py`：`from src.utils.utils import get_parent_module_and_attr` → `from src.utils.model_utils import ...`
- [x] 3.5 `src/data/components/preprocessing.py`：`from src.utils.utils import load_tokenize` → `from src.data.components.tokenization import load_tokenize`
- [x] 3.6 `src/common/modules/embedding_aggregator.py`：`from src.utils.tensor_utils import create_last_k_mask` → `from src.utils.model_utils import create_last_k_mask`
- [x] 3.7 `src/utils/launcher_utils.py`：删除 `from src.utils.utils import has_class_object_inside_list`，将 2 处调用 inline 为 `any(isinstance(cb, cls) for cb in callbacks)`

## 4. 清理旧代码

- [x] 4.1 删除 `src/utils/utils.py`
- [x] 4.2 从 `src/utils/tensor_utils.py` 中删除 `create_last_k_mask` 函数（已迁移至 model_utils.py）

## 5. 验证

- [x] 5.1 `uv run ruff check src/` — 确认无未解析导入和 lint 错误（剩余 5 个为预先存在的 logging_utils/progress_bar 问题，非本次引入）
- [x] 5.2 grep 确认 `from src.utils.utils` 零残留
- [x] 5.3 grep 确认 `from src.utils.custom_hydra_resolvers` 零残留
- [x] 5.4 grep 确认 `create_last_k_mask` 不在 `tensor_utils.py` 中
- [x] 5.5 grep 确认 `has_class_object_inside_list` 不在 `utils.py`（文件已删除）且未作为独立函数存在于任何 utils 模块
- [x] 5.6 smoke check：`uv run python -c "import src.utils.hydra_resolvers; from omegaconf import OmegaConf; ..."` — 确认 `now_tz` resolver 已注册并正确解析（输出 `2026`）
