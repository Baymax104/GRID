## Why
固定混合重排与小型分数修正均未提供可靠净收益。用户授权直接训练CoPMRec decoder的混合概率排序，鉴别候选竞争目标相对额外似然微调的效果。
## What Changes
- 用户采纳高位风险补强：可选training均值上限限制NDCG逐用户归一化分母，保留双teacher保护与恢复直接梯度。
- 复用已核验候选并重读原历史，只训练decoder非共享参数。
- 匹配NLL/指标加权排序两臂，统一双checkpoint audit。
- 配置、手动入口、来源契约、兼容trace与聚焦测试。
## Capabilities
双分支扩展：新增可选content_generation正确关系参照，复用原缓存分数，重复pair取最大间隔，默认content保持旧行为，不改候选或baseline。

2026-10-02再次扩展：用户采纳有条件的内容排序保护。在competitive NDCG训练中提供默认关闭的、基于冻结content正确关系的单侧rank-discount间隔损失；仅training真值筛选保护用户，推理不使用真值，原baseline不变。

### New Capabilities
- `copmrec-decoder-ranking`: 候选内decoder训练与匹配评价。
### Modified Capabilities
无baseline行为变更。
## Impact
新增CoPMRec专用data/model/config/script；共享trace新增schema，旧协议保持。

2026-10-02扩展：依据已有失败样本和初始训练分数导数诊断，加入可选有效竞争pair归一化，旧目标仍为默认。保持同研究问题、baseline与候选边界；假设是容易负例分母相对稀释已有命中的竞争监督，待用匹配训练与最终损益鉴别，尚非性能根因证明。

## 2026-10-03边界辅助监督
用户采纳Top10边界辅助设计，先32 training用户零更新参数探针，通过后实现默认关闭的训练项与training-only尺度校准。保留原CoPMRec目标/候选/基线，匹配szuw834d预算，完整训练用户手动开始。
