## 1. 固定终排实现

- [x] 1.1 为 Liger 添加默认 content 评分和配对 trace hook，保留一次搜索与实际终排 score。
- [x] 1.2 抽取已有完整 SID 混合评分核心，新增 v4 inference-only 模式与原 checkpoint 兼容参数。
- [x] 1.3 严格校验 mixed 协议、候选 rows/SID、两套分数、排名与 TopK。

## 2. 实现验证与运行准备

- [x] 2.1 CPU 测试默认行为、概率恒等式、无标签评分、cold/padding、chunk 映射和严格 trace；回归旧评分核心。
- [x] 2.2 主代理核验 Hydra、单进程命令、OpenSpec、实际源码同步和运行来源。

## 3. 一次固定 evaluation

- [x] 3.1 主代理执行已授权的一次完整 evaluation，独立复算同候选 content/mixed 与匹配 LIGER dense 指标。
- [x] 3.2 依据配对结果更新排序决策，保留证据边界，不自动追加扫描或 Testing。
