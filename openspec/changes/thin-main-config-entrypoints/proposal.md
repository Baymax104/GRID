## Why

当前 `train.yaml` / `inference.yaml` 与 `configs/experiment/*.yaml` 同时暴露 `data_dir`、`ckpt_path` 这类实验级手动输入字段，容易让用户困惑到底应该修改哪一层。随着 experiment 已经成为仓库实际运行的主入口，这些字段继续停留在主入口配置中，只会模糊“导入层”和“实验层”的职责边界。

现在需要把 `train.yaml` / `inference.yaml` 进一步收敛为更薄的导入层，让实验级手动输入统一由 `configs/experiment/*.yaml` 提供，主入口只负责 defaults 组合和通用运行开关。

## What Changes

- 从 `configs/train.yaml` 与 `configs/inference.yaml` 中移除实验级手动输入字段，例如 `data_dir`、`ckpt_path`。
- 保持 `configs/paths/default.yaml` 通过 `${data_dir}` 透传 experiment 顶层输入。
- 保持代码中对 `cfg.get("ckpt_path")` 等访问方式不变，使主入口变薄但不破坏运行时兼容性。
- 通过注释或结构调整明确：experiment 才是手动输入的唯一主入口。

## Capabilities

### New Capabilities
- `thin-main-config-entrypoints`: 让 train/inference 只承担 defaults 导入层职责，实验级手动输入统一下沉到 experiment 配置。

### Modified Capabilities

## Impact

- 受影响文件预计包括：`configs/train.yaml`、`configs/inference.yaml`，以及必要的说明性配置注释
- 不涉及训练/推理代码逻辑变更
- 可能影响“不带 experiment 直接裸跑”的配置完整性预期，但与当前实际使用方式更一致
