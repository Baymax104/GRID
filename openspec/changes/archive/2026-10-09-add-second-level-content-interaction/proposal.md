## Why

A（`token_content_init`）相对匹配随机初始化已有 Beauty seed42 的实际收益，但同一个二层 SID 码仍共享输入表示。本地固定目录分析显示，父码与当前码的加性拟合之外仍存在内容交互；研究问题是显式提供该交互能否在相同训练、搜索协议下进一步改善 A，而不是将重建误差当作推荐瓶颈的证明。

## What Changes

- 提议增加可关闭的第二层固定内容交互输入分支，保持原 A 初始化、独立输出头、合法 CE 与 beam10。
- 首个候选仅学习一个共享标量，从零贡献开始；固定目录残差不训练，不引入投影矩阵、逐物品参数或辅助 loss。
- encoder 历史 SID、teacher forcing 与 CGBS 实际 beam 路径使用一致的前缀表示契约，禁止目标泄漏。
- 提供 interaction、additive、shuffled 三种同规模固定输入对照；完整实验仍由用户手动启动。
- 记录拟合收敛、目录及残差 hash、checkpoint 身份、实际新增参数与训练/推理成本。
- 本次交付研究依据和待实施提案，不修改生产实现、不启动实验。

## Capabilities

### New Capabilities

- `second-level-content-interaction`：固定目录二层加性交互分解、零起点输入注入、因果一致训练推理和有限实验契约。

### Modified Capabilities

无。原 A 默认行为和历史 checkpoint 加载保持原契约。

## Impact

预计涉及 `src/recommendation/tiger_catalog_grounded/` 的可选领域组件，以及 TIGER 输入 embedding 接口、相应 model config、根脚本和聚焦测试。Artifact 读取、lineage 与 writer 继续使用现有公共设施，无新增依赖。

本方法与 PrefixMem、ReSID、条件 memory 相关工作相邻，尚未建立新颖性；候选成功也不能直接宣称首次前缀条件表示。研究证据、详细设计和实验门槛见同目录 `research-assessment.md`、`design.md`、`experiment-plan.md`。
