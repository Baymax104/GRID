## 1. 瘦身主入口配置

- [x] 1.1 从 `configs/train.yaml` 中移除 experiment 级手动输入字段（如 `data_dir`、`ckpt_path`）
- [x] 1.2 从 `configs/inference.yaml` 中移除 experiment 级手动输入字段（如 `data_dir`、`ckpt_path`）

## 2. 保持 experiment 与默认层联动清晰

- [x] 2.1 复核 `configs/experiment/*.yaml` 继续作为 `data_dir` / `ckpt_path` 等字段的唯一手动输入入口
- [x] 2.2 保持 `configs/paths/default.yaml` 通过 `${data_dir}` 透传 experiment 顶层输入
- [x] 2.3 在主入口配置中补充说明性注释，明确 experiment 才是手动输入主入口

## 3. 验证

- [x] 3.1 验证官方 experiment 与 `train.yaml` / `inference.yaml` 组合后 YAML 仍可解析
- [x] 3.2 验证主链路代码对 `ckpt_path` 的消费方式不依赖主入口层显式声明
