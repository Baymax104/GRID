## 1. 逐层诊断

- [x] 1.1 实现不改变 logits 的目标路径 observer，记录 mass/max 分散度、后代数、局部 rank/margin 和实际父前缀存活
- [x] 1.2 将 observer 接入联合 LIGER 预测，输出独立 `liger_dispersion_v1` payload，并保持旧 checkpoint/默认行为兼容

## 2. 产物与分析

- [x] 2.1 实现诊断 bundle validator、writer 和描述性 summary
- [x] 2.2 实现 mass/max bundle 配对分析，包含首次淘汰、恢复方向、后代数分层、配对区间和证据边界

## 3. 配置与交付

- [x] 3.1 增加薄实验配置与人工 launcher 参数，支持 notes、dry-run 和额外 Hydra override
- [x] 3.2 编写冻结协议和手动运行命令，更新研究状态为“已准备、未运行”

## 4. 验证

- [x] 4.1 添加数学、label isolation、候选等价、schema 合并和配对分析的聚焦回归测试
- [x] 4.2 运行聚焦 pytest、Ruff、Hydra/launcher dry-run、shell 语法检查和 `openspec validate --strict`
