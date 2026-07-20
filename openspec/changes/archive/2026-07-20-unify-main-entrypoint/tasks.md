## 1. 统一入口与主配置

- [x] 1.1 新增统一 Python 入口 `src/main.py`，收拢 train / inference 共有装配逻辑
- [x] 1.2 将 `train.yaml` / `inference.yaml` 统一为单一主配置文件
- [x] 1.3 删除旧入口 `src/train.py` 与 `src/inference.py`

## 2. 将模式决策下沉到 experiment

- [x] 2.1 为所有 official `configs/experiment/*.yaml` 显式添加 `run_mode: train|inference`
- [x] 2.2 将 `task_name` 明确保留在各 experiment 顶层
- [x] 2.3 在统一入口中根据 `cfg.run_mode` 分发到 `trainer.fit()` / `trainer.test()` / `trainer.predict()`

## 3. 脚本与文档对齐

- [x] 3.1 将仓库根目录默认 `*.sh` 脚本切换到 `-m src.main`
- [x] 3.2 更新 `AGENTS.md` 与 `README.md` 中的默认运行入口说明

## 4. 验证

- [x] 4.1 验证官方 train experiments 与统一主配置组合后 YAML 仍可解析
- [x] 4.2 验证官方 inference experiments 与统一主配置组合后 YAML 仍可解析
- [x] 4.3 验证统一入口对 `--dry-run`、`ckpt_path`、`test` 的消费语义保持正确
