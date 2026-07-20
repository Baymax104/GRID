## Context

`src/utils/` 包含 16 个文件，其中 2 个文件（`restart_job.py`、`restart_job_utils.py`）已在 docstring 中标注 "Deprecated" 且全仓库无任何引用；5 个文件中存在共 17 个无调用者的函数/resolver。这些死代码还引入了 `psutil`、`pyarrow` 等未在 `pyproject.toml` 中声明的幽灵依赖。

调用者追踪已覆盖三类引用源：
1. Python 导入（`from src.utils.* import ...`）
2. Hydra 配置 `_target_` 引用（`configs/**/*.yaml`）
3. OmegaConf resolver 引用（`${resolver_name:...}`）

## Goals / Non-Goals

**Goals:**

- 删除 2 个已废弃且无引用的整文件
- 删除 5 个文件中 17 个无调用者的函数/resolver
- 清理由函数删除产生的孤立 import 语句
- 消除 `psutil`、`pyarrow` 的幽灵依赖

**Non-Goals:**

- 不拆分 `utils.py` 的 catch-all 结构（属后续独立变更）
- 不为存留函数补充测试
- 不修改任何 YAML 配置文件
- 不改变任何活跃代码的行为

## Decisions

### D1: 直接删除废弃文件，不保留 stub

**选择**：删除 `restart_job.py` 和 `restart_job_utils.py` 整个文件。

**理由**：两个文件全部符号均标注 "Deprecated"，无 Python 导入、无 YAML `_target_` 引用。git 历史保留完整可追溯性，无需在代码库中保留 stub。

**替代方案**：保留文件但添加 deprecation warning → 拒绝，因为它们已经是废弃状态且无人调用，保留只会继续误导维护者。

### D2: custom_hydra_resolvers 仅保留 now_tz

**选择**：删除 6 个未使用 resolver，仅保留 `now_of_timezone`（注册名 `now_tz`）。

**理由**：`now_tz` 在 8 个 YAML 配置中被引用；其余 6 个 resolver 在所有配置中均无引用。保留无引用的 resolver 注册会增加 OmegaConf 全局状态噪音。

### D3: 删除函数后清理孤立 import

**选择**：每次删除函数后，检查并删除因此变为多余的 import 语句。

**理由**：项目使用 Ruff (E/F/I 规则)，未清理的孤立 import 会在 lint 检查中报错（F401）。具体清理项：
- `custom_hydra_resolvers.py`：`ast`、`operator as op`、`DictConfig`、`ListConfig`
- `utils.py`：`import torch.nn.functional as F`
- `file_utils.py`：`from pyarrow import fs as pyarrow_fs`
- `logging_utils.py`：`import json`

**注意**：`file_utils.py` 的 `from shutil import SameFileError` 不清理 — 它被 `copy_to_remote` 使用（行 30）。

### D4: 不删除 decorators.py 的 timeout/retry

**选择**：保留 `decorators.py` 全部内容不变。

**理由**：`timeout` 被 `retry` 内部调用，`retry` 被 `file_utils.py`、`inference_utils.py`、`data/components/readers.py` 使用。虽然 `timeout` 使用 `signal.SIGALRM`（Unix-only），但当前项目训练/推理均在 Linux 上运行，跨平台问题属独立议题。

## Risks / Trade-offs

- **[风险] 误判死代码** → 已通过三类引用源交叉验证（Python 导入 + YAML `_target_` + OmegaConf resolver）。每个被删符号均有明确的"零调用者"证据。
- **[风险] 外部脚本引用被删符号** → 仓库内无 `tests/`，无脚本目录；`restart_job` 系列从未出现在任何 `.sh` 启动脚本中。外部消费者不在本项目 spec 覆盖范围内。
- **[权衡] 删除后若将来需要 restart 功能，需从 git 历史恢复** → 可接受。当前代码已标注 Deprecated，说明团队已决定不再使用该机制。
