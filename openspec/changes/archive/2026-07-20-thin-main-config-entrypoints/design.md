## Context

当前 `train.yaml` / `inference.yaml` 与 `configs/experiment/*.yaml` 同时暴露 `data_dir`、`ckpt_path` 等实验级手动输入字段，而实际运行方式又几乎总是以 `experiment=...` 为主入口。这导致主入口配置和 experiment 配置都像“可编辑入口”，模糊了各自职责。

从代码消费点看，运行时对 `ckpt_path` 的访问主要通过 `cfg.get("ckpt_path")` 完成，而 `paths.default.data_dir` 也已经通过 `${data_dir}` 透传。这意味着主入口配置可以进一步收敛为“defaults 导入层 + 通用运行开关”，把实验级手动输入统一下沉到 experiment。

## Goals / Non-Goals

**Goals:**
- 让 `train.yaml` / `inference.yaml` 只承担 defaults 组合与通用运行开关职责。
- 将 `data_dir`、`ckpt_path` 这类实验级手动输入统一留在 `configs/experiment/*.yaml`。
- 保持当前官方 experiment 的运行行为不变。

**Non-Goals:**
- 不保证不带 experiment 的裸跑配置仍然完整可用。
- 不调整训练/推理 Python 代码中的 `cfg.get(...)` 使用方式。
- 不回收其他 experiment 顶层参数。

## Decisions

### 1. 主入口配置收缩为薄导入层
- 决策：从 `configs/train.yaml` / `configs/inference.yaml` 中移除 `data_dir` 与 `ckpt_path`。
- 原因：这两个字段在当前仓库语义上属于 experiment 级输入，而不是全局入口层职责。

### 2. 继续通过 `paths.default` 透传数据目录
- 决策：保留 `configs/paths/default.yaml` 中 `data_dir: ${data_dir}` 的透传方式。
- 原因：这样 experiment 只需在顶层提供一次 `data_dir`，路径层仍可无缝消费。

### 3. 接受“裸跑主入口更不完整”的 trade-off
- 决策：明确接受 `uv run -m src.train` / `src.inference` 不带 experiment 时不再具备完整手动输入接口。
- 原因：这与当前仓库实际运行方式一致，且能减少用户的配置入口困惑。

## Risks / Trade-offs

- [有人依赖主入口直接填写 `ckpt_path` / `data_dir`] → 需要通过注释与提案说明明确：experiment 才是手动输入主入口。
- [Hydra 合成后缺少字段] → 当前官方 experiment 全部已提供这些字段，且代码通过 `cfg.get(...)` 访问 `ckpt_path`，风险可控。

## Migration Plan

1. 从 `train.yaml` / `inference.yaml` 移除 `data_dir` 与 `ckpt_path`。
2. 保持 experiment 顶层字段不变。
3. 验证官方 experiment 仍能完成 YAML 合成与主链路消费。

## Open Questions

- 当前无阻塞性开放问题。
