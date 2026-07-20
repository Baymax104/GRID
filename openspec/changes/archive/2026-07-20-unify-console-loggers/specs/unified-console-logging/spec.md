## ADDED Requirements

### Requirement: src/ 下所有控制台日志 SHALL 通过 RankedLogger 发出
所有 `src/` 下的 Python 模块，其控制台日志调用 MUST 使用 `RankedLogger(__name__, rank_zero_only=True)` 实例，变量名统一为 `logger`。MUST NOT 直接调用 `logging.info/warning/error/debug()` 或创建独立的 `logging.getLogger()` logger。

#### Scenario: 裸 logging 调用不存在
- **WHEN** 检查 `src/` 下任意 `.py` 文件
- **THEN** 不得出现 `logging.info(` / `logging.warning(` / `logging.error(` 等裸调用（`pylogger.py` 内部实现除外）

#### Scenario: 独立 logger 不存在
- **WHEN** 检查 `src/` 下任意 `.py` 文件
- **THEN** 不得出现 `logging.getLogger(`（`pylogger.py` 内部实现除外）

#### Scenario: RankedLogger 变量名统一
- **WHEN** 检查 `src/` 下任意 `.py` 文件中的 RankedLogger 实例化
- **THEN** 变量名 MUST 为 `logger`

#### Scenario: 多 GPU 环境下仅 rank 0 打印
- **WHEN** 在多 GPU 环境下运行任意实验
- **THEN** 通用信息日志 MUST 仅在 rank 0 上输出，且附带 rank 前缀
