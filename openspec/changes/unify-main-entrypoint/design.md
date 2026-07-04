## Context

当前仓库在运行层面仍然分成两套入口：`src/train.py` + `train.yaml`，以及 `src/inference.py` + `inference.yaml`。虽然最近已经将 experiment 级手动输入逐步下沉到 `configs/experiment/*.yaml`，但训练/推理模式仍然分别由不同 Python 文件与不同主配置文件承载，导致 experiment 的语义与入口语义仍然分裂。

另一方面，这两个 Python 入口已经高度相似：都使用同样的 root setup、resolver 注册、dry-run CLI 改写、`extras(cfg)` 和 `pipeline_launcher(cfg)`。它们真正的差异只剩下：训练路径会执行 `trainer.fit()` / 可选 `trainer.test()`，推理路径会执行 `trainer.predict()`。

## Goals / Non-Goals

**Goals:**
- 用单一 Python 入口 `src/main.py` 统一 train / inference 主链路。
- 用单一主配置文件承载 defaults 导入层与通用运行开关。
- 让每个 official experiment 显式声明自己的 `run_mode` 与 `task_name`。
- 删除旧入口 `src/train.py` 与 `src/inference.py`，避免继续保留双入口语义。

**Non-Goals:**
- 不改变 experiment 内部模型、数据、callback、logger 结构。
- 不回退到由主入口推断 train/inference 模式。
- 不保留 `src/train.py` / `src/inference.py` 的兼容壳。

## Decisions

### 1. 使用显式 `run_mode`，不做隐式推断
- 决策：每个 experiment 顶层新增显式字段 `run_mode: train|inference`。
- 原因：统一入口后必须明确选择 `fit/test` 还是 `predict`，隐式推断（根据 `ckpt_path`、`callbacks`、`train/test` 字段等）都太脆弱。
- 备选方案：靠配置形状推断模式。未采用，因为会让行为隐蔽且容易误判。

### 2. 引入单一 Python 入口 `src/main.py`
- 决策：新增 `src/main.py` 作为唯一运行入口，内部根据 `cfg.run_mode` 分发到训练或推理链路。
- 原因：既然模式由 experiment 决定，入口也应统一，避免 `-m src.train` 与 `run_mode: inference` 之类语义冲突。
- 备选方案：仅统一主配置文件，但保留两个 Python 入口。未采用，因为会继续保留入口语义分叉。

### 3. 统一主配置文件，删除 `train.yaml` / `inference.yaml`
- 决策：用单一主配置文件承载 defaults 层与共有运行字段，并删除旧的两个主入口配置。
- 原因：主入口层不再表达 train 与 inference 差异；这些差异改由 experiment 的 `run_mode`、`task_name` 与其局部配置决定。
- 备选方案：保留两个极薄配置文件。未采用，因为用户目标就是进一步统一主入口。

### 4. 训练链路保留 `test` 作为 train-only 语义
- 决策：统一入口中，只有 `run_mode: train` 会考虑 `cfg.get("test")` 并执行 `trainer.test()`；`run_mode: inference` 只执行 `trainer.predict()`。
- 原因：避免在统一入口下混淆 train-only 与 inference-only 行为。

## Risks / Trade-offs

- [脚本与文档的大量入口引用需要同步更新] → 统一在本次改动中一起替换为 `-m src.main`。
- [某些 experiment 遗漏 `run_mode`] → 在实现中为缺失值提供明确错误，而不是隐式回退。
- [删除旧入口会打断历史命令] → 这是用户显式接受的 breaking change，应在提案和文档中明确说明。

## Migration Plan

1. 新增单一主配置与 `src/main.py`。
2. 为所有 official experiments 增加显式 `run_mode`。
3. 将 train / inference 共有逻辑收拢到单入口，并删除旧入口文件。
4. 更新脚本、README、AGENTS.md 中的命令示例到 `-m src.main`。
5. 做最小静态与配置组合验证，确保官方 experiments 仍能装配。

## Open Questions

- 当前无阻塞性开放问题。
