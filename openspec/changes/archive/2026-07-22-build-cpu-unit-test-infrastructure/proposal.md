## Why

当前仓库没有入库的 `tests/` 目录，也没有 repo-local pytest 依赖与配置，导致后续变更缺少统一的单元测试落点。需要先建立轻量、CPU-only 的测试基础设施，并明确单元测试不得运行完整实验链路。

## What Changes

- 新增 pytest 作为开发依赖，并在 `pyproject.toml` 声明 pytest 发现与严格配置。
- 创建 `tests/` 目录结构作为后续单元测试的统一位置，但本变更不新增实际测试用例文件。
- 在 `AGENTS.md` 增加测试约束说明：单元测试只覆盖无需 GPU 的函数级/模块级逻辑，禁止直接运行完整 experiment。
- 新增测试基础设施规范，约束 pytest 入口、目录结构、CPU-only 单元测试范围与禁止事项。

## Capabilities

### New Capabilities
- `cpu-unit-test-infrastructure`: 定义 pytest 测试基础设施、目录结构以及 CPU-only 单元测试约束。

### Modified Capabilities
- 无。

## Impact

- `pyproject.toml`: 新增 pytest dev dependency 与 pytest 配置。
- `uv.lock`: 同步 pytest 依赖锁定信息。
- `tests/`: 新增测试目录结构与必要占位文件。
- `AGENTS.md`: 记录测试运行约定与测试边界。
- 不修改业务代码，不新增实际单元测试文件。
