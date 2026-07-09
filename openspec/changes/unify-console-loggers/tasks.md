## 1. 替换裸 logging 为 RankedLogger（8 个文件）

- [x] 1.1 `src/utils/file_utils.py`：添加 `RankedLogger` import，实例化 `logger = RankedLogger(__name__, rank_zero_only=True)`，将 3 处 `logging.info/warning()` 改为 `logger.info/warning()`
- [x] 1.2 `src/data/datamodules/base.py`：同上，将 `logging.warning()` 改为 `logger.warning()`
- [x] 1.3 `src/recommendation/base_recommender.py`：同上，将 `logging.warning()` 改为 `logger.warning()`
- [x] 1.4 `src/quantization/residual_kmeans.py`：同上，将 3 处 `logging.info()` 改为 `logger.info()`，删除 `import logging`
- [x] 1.5 `src/quantization/residual_vector_quantization.py`：同上
- [x] 1.6 `src/quantization/residual_quantization_vae.py`：同上
- [x] 1.7 `src/utils/inference_utils.py`：删除 `log = logging.getLogger(__name__)`，改为 `RankedLogger` 实例，变量名 `logger`；将 11 处 `log.info/warning()` 改为 `logger.info/warning()`
- [x] 1.8 `src/utils/utils.py`：删除 `import logging`（`extras` 函数中不直接使用 logging）

## 2. 重命名现有 RankedLogger 变量为 `logger`（~10 个文件）

- [x] 2.1 `src/main.py`：`console_logger` → `logger`
- [x] 2.2 `src/utils/launcher_utils.py`：`command_line_logger` → `logger`
- [x] 2.3 `src/utils/logging_utils.py`：`log` → `logger`
- [x] 2.4 `src/utils/utils.py`：`log` → `logger`
- [x] 2.5 `src/utils/rich_utils.py`：`log` → `logger`
- [x] 2.6 `src/utils/instantiators.py`：`log` → `logger`
- [x] 2.7 `src/utils/restart_job.py`：`command_line_logger` → `logger`
- [x] 2.8 `src/utils/restart_job_utils.py`：`command_line_logger` → `logger`
- [x] 2.9 `src/data/components/dataloaders.py`：`command_line_logger` → `logger`
- [x] 2.10 `src/data/components/datasets.py`：`command_line_logger` → `logger`
- [x] 2.11 `src/common/modules/transformer_base_module.py`：`console_logger` → `logger`

## 3. 删除 TensorFlow 残留

- [x] 3.1 `src/main.py`：删除 `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`

## 4. 验证

- [x] 4.1 grep 确认 `src/` 下无 `logging.info(` / `logging.warning(` / `logging.error(`（`pylogger.py` 除外）
- [x] 4.2 grep 确认 `src/` 下无 `logging.getLogger(`（`pylogger.py` 除外）
- [x] 4.3 grep 确认 `src/` 下无 `RankedLogger` 实例用非 `logger` 变量名
- [x] 4.4 grep 确认 `TF_CPP_MIN_LOG_LEVEL` 零残留
- [x] 4.5 import 检查：所有改动文件 import 通过
