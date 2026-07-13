## Why

`src/utils/utils.py` 职责分裂，混合了启动预处理、模型模块操作、分词辅助三类不相关功能，文件名在 `utils/` 包内语义模糊。同时 `custom_hydra_resolvers.py` 因 commit `f8113d1` 误删 import 成为 dead code，`now_tz` resolver 实际未注册；`decorators.py` 的 `timeout` 装饰器依赖 `signal.SIGALRM`，Windows 下不可用。

## What Changes

- 拆分 `src/utils/utils.py` 为两个职责单一的模块：
  - `src/utils/startup.py`：`extras()` + `print_warnings_for_missing_configs()`
  - `src/utils/model_utils.py`：`delete_module` / `find_module_shape` / `reset_parameters` / `get_parent_module_and_attr`
- 将 `load_tokenize` 从 `utils.py` 移至 `src/data/components/tokenization.py`（唯一消费者在 data 域）
- 将 `create_last_k_mask` 从 `tensor_utils.py` 移至 `model_utils.py`（序列掩码工具，与 keyed bundle 协议无关）
- 将 `has_class_object_inside_list`（1 行工具函数，唯一消费者）inline 到 `launcher_utils.py`
- 删除 `src/utils/utils.py`
- **BREAKING**（模块路径）：`custom_hydra_resolvers.py` 重命名为 `hydra_resolvers.py`，并在 `main.py` 恢复显式 import 修复回归 bug
- 修复 `decorators.py` 的 `timeout` 装饰器：Windows 下 `signal.SIGALRM` 不可用时 `raise NotImplementedError`
- 更新 ~8 处外部消费者的导入路径

## Capabilities

### New Capabilities
- `utils-single-responsibility-modules`: utils 包中每个模块 SHALL 保持单一职责，禁止 catch-all 模块；跨域业务逻辑 SHALL 移至对应域而非留在 utils 中

### Modified Capabilities
- `utils-package-static-imports`: spec 中引用的 `utils.py` 已删除，需更新为新的具体子模块名（`startup.py`、`model_utils.py`）

## Impact

- **代码文件**（~10 个）：3 个新建 + 1 个删除 + 1 个重命名 + ~8 个导入路径更新
- **无配置变更**、**无 API 变更**、**无 checkpoint 影响**
- 导入路径变更涉及 `src/main.py`、`src/recommendation/`（3 个文件）、`src/data/components/preprocessing.py`、`src/common/modules/embedding_aggregator.py`、`src/utils/launcher_utils.py`
