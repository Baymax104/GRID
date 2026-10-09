## 1. 缓存模型

- [x] 1.1 实现共享encoder下的mass20/max20/mass30固定搜索与候选池缓存
- [x] 1.2 实现evaluation-only、训练拒绝和beta0输出契约

## 2. Schema与分析

- [x] 2.1 实现缓存schema、精确候选池/分数/来源校验和本地writer
- [x] 2.2 实现固定beta扫描、selection/audit划分、bootstrap与三种决定

## 3. 入口与协议

- [x] 3.1 增加evaluation数据配置、模型/回调/实验配置和人工launcher
- [x] 3.2 更新冻结协议、研究状态和当前计划

## 4. 验证与同步

- [x] 4.1 添加真实三搜索、schema、beta0等价、划分与停止规则测试
- [x] 4.2 运行聚焦pytest、Ruff、Hydra/Bash、OpenSpec strict并同步node1
