## Context

基础 `JointMixtureLiger` 是 CoPMRec v0。当前最终候选排序只用 content，验证默认 dense。用户采纳 research 中的候选条件相关性设计并授权实现；完整实验仍由用户手动开始。

## Goals / Non-Goals

**Goals:** 实现无冻结 v1，共用 encoder/内容映射/decoder 和单 optimizer；v0 兼容，版本可辨认；selection 最终 hybrid NDCG10 选点，audit 只读固定 best；具备可运行配置与聚焦验证。

**Non-Goals:** 不运行正式训练，不复制 teacher，不恢复已退役入口，不改变 LIGER baseline，不声明收益已实现。

## Decisions

1. v1 继承基础模型；提取基础 loss 的带表示 helper，使 v0 保持同一计算/RNG顺序，v1 单次 encoder/内容投影服务所有目标。v0 旧入口、checkpoint 协议继续兼容；新版本入口记录 `copmrec_version`。
2. 相关性 head 为 `[h,v,h*v,d] -> Linear(128) -> GELU -> Linear(1)`，输出层零初始化；d 是当前 decoder 输入 start+完整 SID 后的末位 hidden state。候选 decoder 按64对分片，无冻结特征缓存。
3. training 每微批随机均匀取4例，候选为 mass beam20+cold+content Top20+正例，去重；基础 loss 覆盖完整 batch。排序 CE 权重1，四项均值相加。离散候选 ID 不求导，评分表示保留梯度。
4. hybrid 推理仅 beam+cold，最终用相关性分数；dense 诊断保持 content 定义。复用已有 trace schema，把 hybrid rank/TopK 更新为最终评分并标明版本。
5. 复用固定 selection/audit 原始历史的数据算法，迁入无训练阶段假设的 CoPMRec datamodule，旧 scratch 路径保留兼容别名。两版新开发入口共享 selection/audit 协议，原 v0正式入口仍沿用旧配置。配置支持显式全体/testing评价，不默认扩大新实验。
6. DDP 使用无重复 evaluation 分片，不 broadcast mutable buffers；training保持全局batch256（双卡每卡128），50k更新；v1所有参数参与同一 loss 图。checkpoint 严格区分 v1 与 v0；预训练 v0加载是显式 weights-only 输入，不能当恢复 v1 optimizer。

## Risks / Trade-offs

- decoder对全部cold逐个评分增加计算 → 分片+真实成本探针，不自动修改候选协议。
- 零输出初始化首步 residual上游梯度为零 → 基础目标仍训练，非零后验证新增梯度路径。
- train/inference候选分布差异与共享任务冲突 → 记录契约，最终看Recall/NDCG净收益。
- v0旧checkpoint没有新版本键 → 接受其旧joint协议；v1要求独立标识，禁止静默混用。
- 旧audit重复使用 → 交付仅开发验证，不称独立效果确认。
