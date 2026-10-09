## 1. 训练与概率实现

- [x] 1.1 核对50k基线resolved config和输入指纹，固定协议所需的组件参数
- [x] 1.2 提取可微分合法前缀分布计算，验证枚举值和梯度一致
- [x] 1.3 增加独立联合训练模式及自包含全局bias，原模式默认行为不变
- [x] 1.4 接通L_sid+L_content+L_mix和训练日志，保证cold mask不污染混合分布
- [x] 1.5 接通联合checkpoint恢复和推理，支持预定alpha1消融并拒绝门控覆盖

## 2. 配置与运行契约

- [x] 2.1 新增薄experiment和组件配置，对齐50k预算、dense选点与基线初始化
- [x] 2.2 通过根脚本及src/main.py交付，验证dry-run、notes、错误输入和override透传
- [x] 2.3 提供1次训练及3次prediction手动命令、真实来源引用和顺序依赖

## 3. 聚焦验证与交付

- [x] 3.1 运行合成梯度、概率归一化、训练/解码一致性、无标签泄漏和checkpoint回归测试
- [x] 3.2 完成Hydra compose、Bash语法及真实参数转发检查
- [x] 3.3 完成有限smoke并记录额外时间和显存；不运行完整数据实验
- [x] 3.4 运行OpenSpec strict并更新研究状态，明确实现完成与效果未知
- [x] 3.5 如交付node1，按既有Mutagen协议flush、核对四会话状态及关键文件哈希

## 4. 用户手动实验后的审计

2026-09-24进度：zl9gv56p训练与best48500 checkpoint已核验；c01w22tw/84d7wkgd/oxnmyhqy三臂testing完成并复算。相对原LIGER主门槛通过，固定0.5对照NDCG增量及生成概率必要性未确认；保留整体方法、限定机制主张，本实例结题，剩余预算0。报告见../../../../research/docs/grid-experiments/2026-09-24-liger-joint-testing-result.md。

- [x] 4.1 回收训练及预定评价，核验lineage、用户/标签和配对指标
- [x] 4.2 依主门槛和辅助对照作出保留、收缩或结题决定，不自动追加预算
