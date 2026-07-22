## ADDED Requirements

### Requirement: 项目 SHALL 提供 pytest 测试基础设施
项目 SHALL 在开发依赖中提供 pytest，并在 repo-local 配置中声明标准测试发现入口，使后续变更可以统一通过 pytest 添加和运行测试。

#### Scenario: pytest dependency is declared
- **WHEN** 维护者检查 `pyproject.toml` 的开发依赖配置
- **THEN** `pytest` MUST be declared in the dev dependency group

#### Scenario: pytest discovery is configured
- **WHEN** 维护者检查 `pyproject.toml`
- **THEN** it MUST configure pytest to discover tests from `tests/`
- **AND** it MUST configure standard `test_*.py` file discovery
- **AND** it MUST enable strict pytest configuration and marker validation

### Requirement: 项目 SHALL 提供测试目录结构但不在本变更新增测试用例
项目 SHALL 创建 `tests/` 目录结构作为后续测试文件落点。本基础设施变更 MUST NOT 新增实际 `test_*.py` 测试文件。

#### Scenario: test directories exist
- **WHEN** 维护者检查仓库目录结构
- **THEN** `tests/` MUST exist
- **AND** `tests/unit/` MUST exist for CPU-only unit tests

#### Scenario: no test cases are introduced by this change
- **WHEN** 维护者检查本变更新增文件
- **THEN** it MUST NOT add `test_*.py` files

### Requirement: 单元测试 SHALL 只覆盖无需 GPU 的逻辑
单元测试 SHALL 限定为无需 GPU、无需分布式执行、无需真实训练/推理实验的函数级或模块级验证。

#### Scenario: CPU-only unit test boundary is documented
- **WHEN** 维护者检查 `AGENTS.md`
- **THEN** it MUST state that unit tests are limited to CPU-only logic
- **AND** it MUST state that tests should use minimal in-memory inputs when practical

#### Scenario: full experiments are forbidden in unit tests
- **WHEN** 维护者检查 `AGENTS.md`
- **THEN** it MUST state that unit tests MUST NOT directly run full experiments
- **AND** it MUST forbid invoking `src.main`, `torchrun`, full Hydra experiment execution, or real Trainer training/inference loops from unit tests
