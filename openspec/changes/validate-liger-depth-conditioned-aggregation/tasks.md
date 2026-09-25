## 1. 深度条件聚合

- [x] 1.1 扩展概率混合processor，支持depth0=max、后续depth=mass并保持既有模式兼容
- [x] 1.2 将`max_root_mass_deep`接入联合模型推理控制、训练拒绝和trace metadata

## 2. 结果分析

- [x] 2.1 实现三臂bundle来源核验、覆盖/NDCG/Recall配对区间和冻结停止决定
- [x] 2.2 增加薄实验配置、单臂人工launcher、协议、结果审计脚本和研究状态

## 3. 验证与交付

- [x] 3.1 添加逐层数值等价、默认回归、训练拒绝、schema和停止规则测试
- [x] 3.2 运行聚焦pytest、Ruff、Hydra/Bash预检、OpenSpec strict并同步node1
