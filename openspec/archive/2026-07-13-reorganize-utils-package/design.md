## Context

`src/utils/` 包含 13 个模块，其中 `utils.py` 是历史积累的 catch-all 文件，混合了启动预处理（`extras`）、模型模块操作（`delete_module` 等）、分词辅助（`load_tokenize`）三类不相关功能。此外存在两个已知问题：

1. `custom_hydra_resolvers.py` 在 commit `f8113d1`（统一日志重构）中被 ruff 误判为 unused import 而删除了 `main.py` 中的显式导入，导致 `now_tz` resolver 未注册。git 历史中 `0ec29ed` 曾修复过同一类问题。
2. `decorators.py` 的 `timeout` 装饰器使用 `signal.SIGALRM` / `signal.alarm`，仅 Unix 可用。当前所有 `@retry()` 调用均无参，不触发 timeout 路径，但留有隐患。

openspec `utils-package-static-imports` 规约要求：不使用动态导出、调用方从具体子模块导入、内部模块直接导入兄弟。本次重构必须保持这些约束。

## Goals / Non-Goals

**Goals:**
- 消除 `utils.py` 的职责分裂，每个模块单一职责
- 修复 `custom_hydra_resolvers.py` 回归 bug（恢复 `now_tz` resolver 注册）
- 修复 `decorators.py` Windows 兼容性
- 将 `create_last_k_mask` 从 `tensor_utils.py` 归位到模型工具域
- 将 `load_tokenize` 从 utils 移至 data 域

**Non-Goals:**
- 不做子包化重组（`utils/logging/`、`utils/io/` 等）— 13 个文件中 6 个只有 0-1 个外部消费者，分组收益不抵 blast radius
- 不修改 `pylogger.py`（9 个外部消费者，改路径纯属噪声）
- 不改变任何运行时行为
- 不修改 `file_utils.py` / `inference_utils.py` / `tensor_utils.py`（keyed bundle 部分）/ `rich_utils.py` / `progress_bar.py` / `cli_utils.py` / `logging_utils.py` 的内容

## Decisions

### 决策 1：拆分 `utils.py` 为 `startup.py` + `model_utils.py`（而非移至各自域）

**选择**：在 utils 包内新建两个文件，而非将函数移至调用方所在域。

**理由**：`extras()` 是通用启动逻辑（不属于 main.py 的业务），模型操作函数被 recommendation 域的 3 个文件共用（不属于任何一个文件）。留在 utils 内保持中性工具定位。

**备选方案**：将 `extras` 移至 `launcher_utils.py`（它已是启动入口）。排除——`launcher_utils` 职责是组件实例化，`extras` 是配置预处理，语义不同。

### 决策 2：`load_tokenize` 移至 `src/data/components/tokenization.py`

**选择**：跨域移动到 data 域。

**理由**：唯一消费者是 `src/data/components/preprocessing.py`，函数依赖 `transformers` tokenizer，属于 data 域业务逻辑。utils 不应承载域特定逻辑。

### 决策 3：`create_last_k_mask` 移至 `model_utils.py`

**选择**：与 `delete_module` 等模型操作工具同放。

**理由**：纯 tensor 操作，消费者是 `src/common/modules/embedding_aggregator.py`。与 keyed bundle 协议无关，不应留在 `tensor_utils.py`。

**备选方案**：移至 `src/common/` 域。排除——common 当前无 utils 文件，为一个函数新建域级工具文件过度。

### 决策 4：`has_class_object_inside_list` inline 到 `launcher_utils.py`

**选择**：删除独立函数，在唯一调用点 inline 为 `any(isinstance(cb, cls) for cb in callbacks)`。

**理由**：1 行 `any()` 表达式，唯一消费者是 `launcher_utils.py` 的 `initialize_pipeline_modules`。独立函数增加了不必要的间接层。

### 决策 5：`custom_hydra_resolvers.py` 重命名为 `hydra_resolvers.py` 并在 `main.py` 恢复 import

**选择**：重命名 + 恢复 `import src.utils.hydra_resolvers  # noqa: F401`。

**理由**：`custom_` 前缀冗余（Hydra resolver 本身就是"自定义"的）。`# noqa: F401` 防止 ruff 再次误删。import 放在 `main.py` 顶层（Hydra 配置组装前），与 `rootutils.setup_root` 之后。

### 决策 6：`decorators.py` Windows fallback 用 `NotImplementedError`

**选择**：在 `timeout` 装饰器中检测 `signal.SIGALRM` 是否存在，不存在时 `raise NotImplementedError("timeout decorator requires Unix signal.SIGALRM")`。

**理由**：显式失败比静默 no-op 安全。当前无触发路径（所有 `@retry()` 无参），修复是预防性的。未来若有人在 Windows 上传 `fn_execution_timeout_s`，会立即得到明确错误而非静默失效。

**备选方案**：no-op fallback。排除——静默失效可能导致无限等待，更危险。

## Risks / Trade-offs

- **[导入遗漏]** ~8 处外部导入路径变更可能遗漏。→ **缓解**：删除 `utils.py` 后 ruff 会报所有未解析导入；再 grep `from src.utils.utils` 确认零残留。
- **[resolver 循环导入]** 恢复 `hydra_resolvers` import 可能引入循环依赖。→ **缓解**：`hydra_resolvers.py` 仅依赖 `pytz` 和 `omegaconf`，不导入任何 `src.utils` 模块，无循环风险。
- **[重命名增加变更量]** `custom_hydra_resolvers.py` → `hydra_resolvers.py` 增加了 1 处 import 变更。→ **缓解**：该文件无外部消费者，仅 `main.py` 需更新。
