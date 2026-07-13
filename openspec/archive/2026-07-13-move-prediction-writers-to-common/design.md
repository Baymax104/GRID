## Context

`src/utils/inference_utils.py` 定义了两个 Lightning callback：`BaseBufferedWriter`（缓冲积累 + flush 抽象基类）和 `LocalPickleWriter`（分片写 pickle → 主进程合并为 keyed prediction bundle）。它们是推理结果写入的实验组件，通过 Hydra `_target_` 字符串实例化，不被任何 Python 文件直接 import。

该文件已依赖 `src.common.components.model_output.ModelOutput`，且与 `src/common/components/` 中的 `eval_metrics`、`loss_functions`、`scheduler` 同类——都是被 Hydra 配置实例化的实验组件。当前放在 utils 中不符合域归属。

## Goals / Non-Goals

**Goals:**
- 将 `inference_utils.py` 移至 `src/common/components/`，使实验组件归属一致
- 重命名为 `prediction_writers.py` 以匹配 components 目录的功能描述性命名风格（`eval_metrics.py`、`loss_functions.py`、`scheduler.py`）

**Non-Goals:**
- 不修改 `BaseBufferedWriter` / `LocalPickleWriter` 的代码逻辑
- 不修改 utils 包的其他文件
- 不修改 keyed prediction bundle 协议

## Decisions

### 决策 1：目标位置为 `src/common/components/`

**选择**：移至 `src/common/components/prediction_writers.py`。

**理由**：`BaseBufferedWriter` / `LocalPickleWriter` 是被 Hydra `_target_` 实例化的实验组件，与 `common/components/` 中的 `eval_metrics`、`loss_functions`、`model_output`、`scheduler` 同类。移动后对 `model_output.ModelOutput` 的依赖变为同域。

### 决策 2：文件名改为 `prediction_writers.py`

**选择**：不保留 `inference_utils.py` 名称，改为 `prediction_writers.py`。

**理由**：`common/components/` 下的命名风格是功能描述性复数名词（`eval_metrics`、`loss_functions`）。`inference_utils` 是域描述性命名，风格不一致。`prediction_writers` 准确描述内容（prediction writer callbacks）且风格统一。

### 决策 3：仅需更新字符串引用，无需更新 Python import

**选择**：仅更新 4 处 Hydra `_target_` 字符串引用。

**理由**：`inference_utils.py` 不被任何 Python 文件直接 import。引用全部是字符串形式（1 个 Python 字符串 + 3 个 YAML `_target_`），不触发 import 路径变更。blast radius 极小。

## Risks / Trade-offs

- **[Hydra `_target_` 字符串遗漏]** 4 处字符串引用可能遗漏。→ **缓解**：移动后 grep `src.utils.inference_utils` 确认零残留。
- **[checkpoint 兼容性]** 已有的 checkpoint 中可能序列化了 `src.utils.inference_utils.LocalPickleWriter` 的类路径。→ **缓解**：`LocalPickleWriter` 是 callback，不被序列化到 checkpoint 中（checkpoint 仅保存 model state_dict 和 optimizer state），无兼容性风险。
