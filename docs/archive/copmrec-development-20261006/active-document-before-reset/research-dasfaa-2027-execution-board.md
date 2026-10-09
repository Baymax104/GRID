# DASFAA 2027 当前执行看板

> 2026-10-03 实施更新：用户采纳候选相关性组件，基础 CoPMRec 记为 v0，新全参数联合模型 v1 已实现并验证。本地120项测试、node1两进程/生产模型探针通过，详见 [版本与交付](../../GRID/docs/copmrec-versions.md)。完整训练及相对LIGER dense的收益尚未验证，链路效果仍未完成；不重置旧预算或自动启动实验。

> 2026-10-03 后续修正：基础 CoPMRec（BMX-116）结果有效，但相对 LIGER dense 的收益未明确建立，完整方法链路仍待完成。旧冻结收益转化入口已归档清理，108 项测试、OpenSpec strict 与 Mutagen/远端核验通过。当前为 [候选条件相关性组件设计](2026-10-03-copmrec-conversion-redesign.md)，尚未实施，新增完整预算/运行均为 0；下方之前的建议及历史完成项按日期保留，不能视为当前链路已完成。

> 2026-10-03 方法梳理更新：用户确认推荐模型所有参数持续可训练，允许预生成 embedding 与 SID。建议主方法收敛到无 teacher 的内容条件前缀概率联合模型；完整公式、证据及运行边界见[整体方法重梳理](2026-10-03-copmrec-unified-method.md)。本次完成设计记录与研究状态同步，生产入口尚未迁移，新完整预算为 0，既有运行未停止或替换。以下按原日期保留的完成项和缺口不是当前结果的重新核验。

更新：2026-09-25。事实源为 `research-state.yaml`；主方法为 **CoPMRec**，主baseline为 **LIGER (GRID adapted v1, 50k)**。

## 已完成

- 推荐模型从零整体训练、统一概率混合训练和解码实现；上游SID和内容向量固定。
- Beauty/seed42匹配50k主比较及固定0.5、同checkpoint alpha1对照；trace/summary/来源核验完成。
- CoPMRec相对LIGER主门槛通过；保留NDCG增量和生成概率必要性未确认的结果。
- 方法命名、固定研究问题、主张—证据审查、英文工作稿与自动生成结果表。

## 当前论文判断

CoPMRec的论文机制解释已补齐：正文现在从候选遗漏定位到叶节点内容证据与前缀beam决策的粒度错位，以分散内容支持推出子树总质量，并用首次路径分歧诊断连接mass margin与目标路径存活；相对PAG的边界表述为best-descendant value与set-level probability allocation。该链条不改变mass独立最终收益未确认的边界。LaTeX与8页PDF检查通过，未新增实验。

主方法已冻结为仅使用子树总质量`mass`的内容前缀聚合；`max`只保留为机制消融，不参与最终方法。独特价值定位为在有限beam剪枝前，将分散在合法SID子树多个商品上的用户条件内容证据转成规范化前缀概率。该定位有路径分歧诊断支持，但不声称mass相对max已取得独立最终推荐增益。见[方法梳理](grid-experiments/2026-09-25-copmrec-mass-only-method-synthesis.md)。

2×2交叉解码已完成：l6hkjuoh/hs3ex5dl补齐两单元。同alpha下联合训练的NDCG/Recall增量未确认，训练×解码交互未确认；联合checkpoint内alpha1显著优于固定0.5。完整方法总体效果保留，独立训练和协同主张收缩。见[2×2结果](grid-experiments/2026-09-24-liger-training-decoding-factorial-result.md)。该阶段剩余预算0。

同得分机制控制已完成：nvrx2rkd/pc9mo9ld，固定合法支持的内容引导收益得到支持；总质量相对最大后代仅有覆盖优势，下游优势未确认，净增5命中。见[结果报告](grid-experiments/2026-09-24-liger-mechanism-result.md)。新增0训练2预测阶段已耗尽，剩余0。

偏好分散诊断已完成：5wfpsg9a/sjf8qcgs。总质量相对最大后代的margin能以86.49%的准确率预测目标路径分歧方向，且组间margin差区间为正；最终Recall/NDCG差值区间跨0。该结果支持“分散内容支持的前缀路径建模”场景解释，不支持总质量独立提高最终推荐或已观测多兴趣。见[结果报告](grid-experiments/2026-09-25-liger-preference-dispersion-result.md)。本阶段0训练2预测已耗尽。

深度条件转化控制已完成：gwfdimng。固定max-root/mass-deep没有保留相对max的覆盖优势，且相对mass覆盖显著下降；NDCG/Recall优势未确认。分深度关联不能直接拼接为搜索策略，该控制按协议停止，不搜索其他层级计划。见[结果报告](grid-experiments/2026-09-25-liger-depth-conditioned-result.md)。本阶段0训练1预测已耗尽。

可写主线为“候选遗漏问题—内容条件前缀概率—联合训练和解码—覆盖率及推荐质量”。现有证据支持单实例整体效果，不支持通用SOTA、效率优势、动态门控或双分支缺一不可。尚未达到投稿证据完整性。

## 尚缺证据与文稿工作

已建立[论文实验矩阵v1.1](2026-09-25-copmrec-experiment-matrix.md)：seed固定为42/200/2025。W&B已核验Beauty/200 CoPMRec训练`83djht0e`及best step45500，但没有testing；用户报告的seed2025及其他seed200结果待来源审计。当前核心剩余上限至多15训练/36预测，审计后只下调；TIGER/COBRA为条件槽位。矩阵不授予完整运行权限。

| 项目 | 状态 | 对结论的影响 |
|---|---|---|
| 独立seed与跨数据集证据 | 未具备 | 限制稳定性和泛化主张 |
| 独立训练机制增量 | 未确认 | 当前收缩措辞，不追加模块或调参 |
| 同类强方法名单及统一比较协议 | 矩阵已列层次，执行协议待冻结 | LIGER为主baseline，TIGER为基础对照，COBRA仍是条件候选 |
| 广泛相关工作、完整复现说明 | 部分完成 | 16项原文机制比较已完成；PAG/RGD/V-STAR纳入近邻，非穷尽查新 |
| 正式会议模板、当届投稿要求及全文审查 | 未完成 | 当前article工作稿不作为投稿版本 |

## 预算与执行边界

整体训练阶段1/1训练、3/3预测完成；偏好分散诊断0训练、2/2预测完成；深度条件转化0训练、1/1预测完成，均剩余0。本次新增完整实验预算0；不会自动启动训练、推理或新审计矩阵。已关闭的动态门控、独立排序与归档路线不恢复。

## 当前文档

- [主方法及论文完整性审查](2026-09-24-copmrec-paper-assessment.md)
- [扩展文献与贡献复评](../literature/2026-09-24-copmrec-content-generative-review.md)
- [英文工作稿](../src/copmrec.tex)
- [当前计划](../ideas/current-plan.md)
- [完整测试核验](grid-experiments/2026-09-24-liger-joint-testing-result.md)

## 历史边界

[2026-09-24更新前看板](dasfaa-2027-execution-board-before-copmrec-20260924.md)原文保留，仅用于追溯。其中“当前有效”“最新有效”及旧预算均是历史文字，不是当前授权或路线依据。
