## 1. Pytest 配置

- [x] 1.1 将 `pytest` 添加到 `pyproject.toml` 的 dev dependency group
- [x] 1.2 在 `pyproject.toml` 添加 pytest 配置，声明 `tests/` 发现路径、`test_*.py` 文件模式、strict config 与 strict markers
- [x] 1.3 使用 `uv` 同步更新 `uv.lock`

## 2. 测试目录与约束文档

- [x] 2.1 创建 `tests/` 与 `tests/unit/` 目录结构，并添加必要占位文件以便目录入库
- [x] 2.2 确认本变更不新增实际 `test_*.py` 测试文件
- [x] 2.3 在仓库根 `AGENTS.md` 增加测试运行约定与 CPU-only 单元测试边界
- [x] 2.4 在 `AGENTS.md` 明确单元测试不得直接运行完整 experiment、`src.main`、`torchrun`、完整 Hydra experiment 或真实 Trainer 训练/推理链路

## 3. 验证

- [x] 3.1 运行 pytest 配置级 smoke，确认 pytest 可启动且配置可加载
- [x] 3.2 运行 `openspec validate build-cpu-unit-test-infrastructure --strict`
- [x] 3.3 检查 diff，确认业务代码未被修改且没有新增测试用例文件
