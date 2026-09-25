## Why

内容初始化已有正向结果，但历史 Full 没有包含该初始化，也没有辅助监督单独作用的对照。目前无法判断剪枝前的内容评分是否解决了初始化之后的剩余错误，需要一次有结束条件的机制验证。

## What Changes

- 新增同初始化的 `content_init_aux`（B）和 `content_init_full`（C）；旧八条件的计算和 checkpoint 契约保持不变。
- 提供仅用于推理的评分关闭、精确内容聚合、保持分支大小的内容置乱及候选后重排干预；通过已有预测与 prefix trace writer 输出逐用户证据。
- 新增只顺序训练 Beauty B/C 的手动队列，固定物理 GPU 0、1，保留原 20k steps / batch 256 / seed42 协议及额外 override。
- 固定假设、对照、证据链和去留规则；不把代码测试、局部 rank 或单种子开发集结果当作机制确认。

## Capabilities

### New Capabilities

- `cgbs-mechanism-validation`: 同初始化对照与冻结 checkpoint 内容评分干预。

### Modified Capabilities

无。

## Impact

影响 CGBS model/catalog、对应 component config、根目录 launcher 和聚焦测试。复用统一入口与现有 writer，不增加依赖，不修改历史实验产物或基线默认行为。完整 GPU 实验由用户手动启动。
