## Context

仓库当前没有入库的 `tests/` 目录，也没有 pytest 依赖和 pytest 配置。既有验证主要依赖手写 smoke command、`compileall`、`ruff` 与 OpenSpec validate；这些验证适合变更时临时执行，但缺少后续可持续积累的单元测试入口。

本变更只建立测试基础设施，不编写具体测试用例。测试边界需要写入 `AGENTS.md`，让后续 agent 和维护者在新增测试时遵守：单元测试只覆盖无需 GPU 的函数级/模块级逻辑，不能直接运行完整实验。

## Goals / Non-Goals

**Goals:**
- 为仓库建立 pytest dev dependency 与 repo-local pytest 配置。
- 创建 `tests/` 目录结构，作为后续测试文件的统一落点。
- 在 `AGENTS.md` 中记录测试运行入口和单元测试约束。
- 用 OpenSpec 固化 CPU-only 单元测试基础设施要求。

**Non-Goals:**
- 不新增实际 `test_*.py` 测试文件。
- 不改业务代码。
- 不引入 GPU、分布式训练、真实数据目录或外部服务相关测试。
- 不在测试体系中直接运行 `src.main`、`torchrun` 或完整 Hydra experiment。

## Decisions

### 使用 pytest 作为唯一测试入口

决策：将 `pytest` 加入 `pyproject.toml` 的 dev dependency group，并在 `pyproject.toml` 添加 `[tool.pytest.ini_options]`。

理由：项目已经使用 `uv` 管理依赖，`pyproject.toml` 是当前 Windows 最小依赖清单；pytest 配置放在同一文件能提供明确的 repo-local 标准入口。

备选方案：单独新增 `pytest.ini`。未采用，因为当前项目工具配置已集中在 `pyproject.toml`。

### 建立空测试目录但不新增测试用例

决策：创建 `tests/` 和 `tests/unit/`，通过占位文件保证目录可入库。

理由：用户明确要求本变更只构建基础设施，后续变更需要测试时再编写测试文件；占位目录提供稳定位置，同时避免引入与本变更无关的测试覆盖讨论。

备选方案：立即为数据 preprocessing/collate 添加第一批测试。未采用，因为超出当前变更范围。

### 单元测试约束写入 AGENTS.md

决策：在仓库根 `AGENTS.md` 中新增测试约束，明确单元测试必须 CPU-only，禁止运行完整 experiment。

理由：`AGENTS.md` 是本仓库 agent 操作约定入口，能直接约束后续实现测试的行为。

备选方案：仅写 OpenSpec。未采用，因为 OpenSpec 记录产品/工程契约，`AGENTS.md` 更适合作为日常执行规则。

## Risks / Trade-offs

- [Risk] 空测试目录运行 pytest 时可能显示没有测试。→ Mitigation：本变更只验证 pytest 可被调用和配置可加载；后续新增测试文件时再要求具体测试通过。
- [Risk] pytest 依赖更新可能触发 lock 文件变化。→ Mitigation：通过 `uv` 更新依赖并提交 `uv.lock` 的受控变更。
- [Risk] 后续测试可能误用完整实验作为单元测试。→ Mitigation：在 `AGENTS.md` 与 spec 中明确禁止事项，并建议只测试 CPU-only 函数级/模块级逻辑。
