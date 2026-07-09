## Context

项目有完善的 `RankedLogger`（`src/utils/pylogger.py`），提供 rank 前缀和 `rank_zero_only` 过滤。但 8 个文件 25 处日志调用绕过它，直接使用裸 `logging.*` 或独立 logger。项目已无 TensorFlow 依赖，`TF_CPP_MIN_LOG_LEVEL=3` 为残留。

## Goals / Non-Goals

**Goals:**
- 所有 `src/` 下日志调用统一经 `RankedLogger`，变量名统一为 `logger`
- 删除 `src/main.py` 中 `TF_CPP_MIN_LOG_LEVEL=3`

**Non-Goals:**
- 不修改 `pylogger.py`
- 不改变 `decorators.py` / `dataloaders.py` 的 `rank_zero_only=False` 行为
- 不修改 `rich_utils.py` 的 `rich.print`
- 不改动第三方库日志配置

## Decisions

### 决策 1：变量名统一为 `logger`
**选择**：所有 RankedLogger 实例变量名统一为 `logger`，无论原有名称是 `log`、`console_logger` 还是 `command_line_logger`。

**理由**：消除命名分歧，新人无需记忆 3 种命名约定。`logger` 是 Python 社区通用命名。

### 决策 2：新建 logger 采用 `rank_zero_only=True`
**选择**：对当前使用裸 logging 的 6 个文件（file_utils、base.py、base_recommender、3 个 quantization），新建的 RankedLogger 实例使用 `rank_zero_only=True`。

**理由**：这些文件中的日志调用都是通用信息，无调试用途，应与项目其余 11 个实例保持一致。

### 决策 3：inference_utils.py 改 RankedLogger
**选择**：删除 `inference_utils.py` 中的 `log = logging.getLogger(__name__)`，改为 `RankedLogger(__name__, rank_zero_only=True)`。

**理由**：该文件的日志调用（推理进度、合并信息）应在 rank 0 独占输出。原独立 logger 无 rank 过滤和前缀。

## Risks / Trade-offs

- **[名称冲突风险]** 某些文件已有名为 `logger` 的其他用途变量。→ **缓解**：实施前逐文件检查，确认无冲突。
- **[行为变化]** inference_utils.py 的日志从所有 rank 打印变为仅 rank 0 打印。→ **缓解**：该文件日志内容为推理进度/合并信息，rank 0 独占输出是正确行为（与 LocalPickleWriter 等一致）。
