## 1. 实现

- [x] 1.1 添加采样、RNG、显式 positives、批量读取及短 optimizer 轨迹回归，复现原逐样本读取。
- [x] 1.2 应用批量采样输入读取，保持原检查、loss 与 checkpoint 契约。

## 2. 验证

- [x] 2.1 运行 Tokenizer 聚焦测试与 OpenSpec strict。
- [x] 2.2 同步 node1，核验旧真实 checkpoint 的输出/梯度/optimizer 状态与耗时。
- [x] 2.3 更新修复与有限验证记录，注明无正式实验启动。
