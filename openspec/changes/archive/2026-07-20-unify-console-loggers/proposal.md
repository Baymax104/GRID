## Why

项目中 8 个文件 25 处日志调用绕过标准 `RankedLogger`，直接使用裸 `logging.info/warning()` 或独立 logger，导致多 GPU 环境下所有 rank 同时打印、缺少 rank 前缀。此外 RankedLogger 实例变量命名不统一（`log`/`console_logger`/`command_line_logger`），`src/main.py` 残留已无依赖的 `TF_CPP_MIN_LOG_LEVEL=3`。

## What Changes

- 将所有裸 `logging.info/warning()` 和独立 `logging.getLogger()` 调用统一替换为 `RankedLogger(__name__, rank_zero_only=True)` 实例，变量名统一为 `logger`
- 将现有 RankedLogger 实例变量名从 `log`/`console_logger`/`command_line_logger` 统一重命名为 `logger`
- 删除 `src/main.py` 中 `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`（项目已无 TensorFlow 依赖）
- **不修改** `decorators.py` / `dataloaders.py` 的 `rank_zero_only=False`（有意的多 rank 日志）
- **不修改** `rich_utils.py` 的 `rich.print`（Rich 树输出，非日志）
- **不修改** `pylogger.py` 本身

## Capabilities

### New Capabilities
- `unified-console-logging`: 要求 `src/` 下所有控制台日志必须通过 `RankedLogger` 实例发出，变量名统一为 `logger`，确保多 GPU 环境下仅 rank 0 打印且附带 rank 前缀。

### Modified Capabilities
<!-- 无，纯机械替换，不改变行为契约。 -->

## Impact

- **代码文件**（~15 个）：8 个替换裸 logging + ~10 个重命名变量
- **无配置变更**、**无 API 变更**、**无 checkpoint 影响**
