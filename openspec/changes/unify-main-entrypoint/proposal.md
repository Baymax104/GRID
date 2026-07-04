## Why

当前仓库同时维护 `src/train.py`、`src/inference.py` 以及 `train.yaml`、`inference.yaml` 两套主入口，虽然它们已经被收缩得很薄，但仍然让“运行方式由哪里决定”变得分散。既然 experiment 已经成为实际运行主入口，训练/推理模式也应由 experiment 显式决定，而不是继续由不同 Python 入口和主配置文件隐式分流。

现在需要统一 Python 入口与主配置入口，让 experiment 显式声明自身是 train 还是 inference，同时删除旧入口，减少重复与语义分裂。

## What Changes

- 新增统一 Python 入口 `src/main.py`，承接当前 train / inference 的共有装配逻辑。
- 将 `src/train.py` 与 `src/inference.py` 删除，不再保留兼容壳。
- 将 `train.yaml` / `inference.yaml` 统一为单一主配置文件，由 experiment 通过显式字段（如 `run_mode`）决定走训练还是推理链路。
- 要求官方 `configs/experiment/*.yaml` 显式声明运行模式与各自的 `task_name`。
- 对齐脚本、文档和运行命令到统一入口。

## Capabilities

### New Capabilities
- `unified-main-entrypoint`: 使用单一 Python 入口与单一主配置文件运行所有官方 experiments，并由 experiment 显式声明 `run_mode`。

### Modified Capabilities

## Impact

- 受影响代码：`src/train.py`、`src/inference.py`、新增 `src/main.py`、共享装配/执行逻辑
- 受影响配置：主入口配置文件、`configs/experiment/*.yaml`
- 受影响脚本/文档：仓库根目录 `*.sh`、`AGENTS.md`、`README.md`
- **BREAKING**：旧命令 `-m src.train` / `-m src.inference` 将被替换为统一入口命令
