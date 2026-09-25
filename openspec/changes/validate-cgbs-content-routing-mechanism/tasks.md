## 1. 范围与证据

- [x] 1.1 核对现有 CGBS 代码、A 条件配置及缺失证据，固定规格和判定边界。
- [x] 1.2 核对 W&B 基线身份、上游产物和当前配置一致性，形成实验判定文档。

## 2. 实现

- [x] 2.1 实现 B/C 同初始化条件，保持历史 arm 契约。
- [x] 2.2 实现冻结 checkpoint 评分干预、精确聚合和干预身份记录。
- [x] 2.3 配置及训练/推理手动队列，保留参数透传和失败停止。

## 3. 验证与交付

- [x] 3.1 通过模型、数值、标签独立性、checkpoint、配置和脚本聚焦测试（125 passed）。
- [x] 3.2 通过 OpenSpec strict、Ruff 与 diff whitespace 检查；未执行真实 GPU 实验。
- [x] 3.3 Mutagen flush 成功，四个 session 均 Watching for changes、无 conflict；交付 0/1 卡训练命令。

## 4. 科学判定（等待手动实验）

- [x] 4.1 完成 B/C 训练，核验 W&B 身份、预算及 best checkpoint（B=z98fozox、C=k6jvoo2v；见 training-outcome.md）。
- [ ] 4.2 分阶段完成必要的冻结干预，连接评分、存活和推荐净收益，给出去留结论。

C screen 已完成：trained=e0t8l1oa、off=sgpj4amu；实际产物配对结果见 `c-screen-outcome.md`。当前推荐点估计向好但区间跨零，C仍未超过A；signal尚待用户手动执行，完整机制资格未确认。

后续C signal已完成：exact=0vis21sg、shuffled_exact=olsg1a90；见 `c-signal-outcome.md`。本轮关闭原型扩容的优先级，将下一候选收敛为状态门控设计。B的逐用户控制与改进版本验证尚未完成，完整机制资格仍未通过；不把下一候选写成已证实结论。
