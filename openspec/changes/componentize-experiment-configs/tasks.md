## 1. 定义并验证 experiment 组件化骨架

- [x] 1.1 为 official experiment 明确统一的单文件骨架约定，包含 `components`、`data_loading`、`model` 等主要域
- [x] 1.2 选择一个 item 类 experiment 和一个 sequence 类 experiment 作为代表，先验证组件化骨架在两类主链路上都可表达

## 2. 重组 item 类 experiment 配置

- [x] 2.1 重组 `sem_embeds_inference.yaml`、`rkmeans_train.yaml`、`rkmeans_inference.yaml` 的主要装配子树到 `components`
- [x] 2.2 重组 `rvq_train.yaml`、`rqvae_train.yaml` 的主要装配子树到 `components`
- [x] 2.3 确保 item 类 experiment 中 `data_loading` 与 `model` 主要保留参数域和组件引用

## 3. 重组 sequence 类 experiment 配置

- [x] 3.1 重组 `tiger_train.yaml` 的 dataloader / dataset / label / model 主要组件到 `components`
- [x] 3.2 重组 `tiger_inference.yaml` 的主要装配子树到 `components`
- [x] 3.3 确保 sequence 类 experiment 中保留少量必要 inline `_target_`，避免过度抽象

## 4. 验证与收尾

- [x] 4.1 复核所有 official experiment 的主链路在文件顶部可快速定位主要组件
- [x] 4.2 做最小配置 smoke check，确认 launcher 顶层 instantiate 入口仍可解析 componentized experiments
- [x] 4.3 清理因骨架重组产生的过时注释，保证配置语义与结构一致
