## Why

fresh 完整 Evaluation 的22363个真实目标均未出现在本人有效历史中，但 LIGER / 固定pool 的错误Top10历史占位为15642 / 19662次，两个固定key组均存在已知命中被历史商品压后的位置损失。这支持一次固定评分下的候选资格检验；训练关系统计未支持新增CF信号方向，已关闭的 residual / mixed / bias / pool 结论与训练预算不恢复。

## What Changes

- 新增共享严格 `apply_history_exclusion(scores, input, catalogmodel)`，仅使用模型输入中有效完整SID，将对应完整目录行设为负无穷，其余原始分数不变；完整目录稳定Top10，cold商品继续可选。
- 新增仅推理的 `HistoryExcludedLiger` 与 `HistoryExcludedFixedLogitPool` 薄子类。前者保留原LIGER checkpoint hook / strict restore，后者复用现有固定pool的双来源与0.5 / 0.5，只在完整 logits 后应用同一个资格helper。旧类与入口默认不变。
- 固定相同有效20商品历史规则、共享 `history_eligibility_contract`、单进程FP32和标准commonwriter。helper不读取labels、raw training或用户key，不构建graph / 新评分 / optimizer。
- 新阶段新增训练0；完整Evaluation最多2次（固定LIGER best45000与当前固定pool均使用同一政策）。只有pool相对**新same-policy LIGER**的R10 / N10均≥10%、两paired差值CI下界正，且同时通过旧基线点阈值，才执行至多2次固定Testing（same-policy LIGER与same pool）。全线程Testing仍最多3次、此前已用1次。
- 按用户在正式运行前的最新直接指示，组件保留与整体目标分别判断：v4.3相对自己冻结旧pool的R10/N10任一相对提升≥3%，另一项点值无退化，可保留继续bad case分析及累计改进，CI和固定分组限制效果主张；未达整体双10%不自动丢弃正向组件。无明显增量或负向则按现有边界收缩。
- 本阶段仍不换history window、pair、weight、temperature或arm，不重置已用5训练 / 30000 steps；其耗尽表示旧登记训练预算耗尽，不是用户禁止未来研究的永久上限。后续成本须依新正向证据、bad case及新明确登记决定，不能自动重置本阶段额度。

## Capabilities

### New Capabilities

- `copmrec-history-eligibility`：基于有效模型输入完整SID的共享历史排除、原checkpoint兼容的单卡推理、same-policy匹配评价及有界晋级。

### Modified Capabilities

无。既有LIGER、v4、v4.1和v4.2的规格及默认行为不改。

## Impact

新增 recommendation helper / 两薄推理子类、model / experiment配置、根推理脚本与聚焦CPU测试；复用统一 `src.main` / Hydra、公共Artifact解析、lineage与commonwriter，无新依赖。历史排除是成熟的资格处理，不宣称原创或官方LIGER bugfix，不能只对CoPMRec排历史并以未排历史LIGER声称方法增量。此处仅规格准备，尚无v4.3正式效果。
