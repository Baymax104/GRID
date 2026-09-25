## 1. 候选构造

- [x] 1.1 抽取可指定processor与beam数的候选生成helper并保持默认行为回归
- [x] 1.2 实现inference-only union20与mass30臂及标准候选trace metadata

## 2. 证据与判定

- [x] 2.1 实现union来源trace、精确并集/覆盖OR校验和Artifact writer
- [x] 2.2 实现三臂来源一致性、配对区间和冻结停止决定

## 3. 入口与协议

- [x] 3.1 增加薄Hydra配置、双臂人工launcher和Bash/Hydra参数校验
- [x] 3.2 更新冻结协议、当前计划与研究状态，记录已消耗testing和预算边界

## 4. 验证与交付

- [x] 4.1 添加候选构造、训练拒绝、schema、比较决策及默认行为回归测试
- [x] 4.2 运行聚焦pytest、Ruff、Hydra/Bash预检、OpenSpec strict并同步node1
