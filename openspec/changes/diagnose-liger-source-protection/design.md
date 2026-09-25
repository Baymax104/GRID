## Context

完成态testing显示union相对mass20新增96、损失81个Top10命中；96个新增全部是max-only目标，80/81个损失来自mass生成目标，另有111个共同命中降序。简单候选增量与收益不单调，且既有learned reranker失败，因此只保留与来源竞争直接对应的单参数保护诊断。

## Goals / Non-Goals

**Goals:**

- 在evaluation上一次性缓存mass20、max20、mass30及共享内容分数。
- 精确复现beta0 union与mass30内容排序，并对mass来源增加标准化分数bonus。
- 用固定网格和确定性selection/audit划分判断是否存在稳健平台。
- 输出机器可读曲线、配对区间、来源门禁和停止决定。

**Non-Goals:**

- 不使用testing调参，不训练模型、门控或reranker。
- 不扫描beam、候选数、特征、网络、quota、seed或数据集。
- 不把evaluation通过称为独立确认或论文效果。

## Decisions

1. **单次三搜索缓存。** 新模型共享一次encoder和内容projection，固定运行mass20、max20、mass30；缓存去重后的`union+cold`与`mass30+cold`候选池、分数和mass成员标记。
2. **本地缓存，不发布W&B Artifact。** 缓存可再生成且包含完整逐用户分数，只在node1运行目录保存；W&B记录config、摘要指标和决定。
3. **按用户标准化内容分数。** 对每个union池用有效候选均值和population std得到z-score，再计算`z+beta*I(mass20)`；beta0必须逐元素复现原内容排序。固定beta为`[0,0.1,0.25,0.5,1.0]`。
4. **确定性50/50开发划分。** 按用户key与seed42确定selection/audit。selection上仅当相邻两个非零beta相对beta0和mass30均有正NDCG点估计且Recall点不降时形成平台；冻结平台中较小beta。audit只评价该beta。
5. **audit门槛。** 冻结beta相对beta0和mass30的NDCG@10配对95%CI下界均须大于0，Recall@10点估计均不降；否则停止。若selection无平台则不查看beta级audit比较并直接停止。
6. **三种结论。** 无平台=`source_protection_no_stable_selection_plateau_stop`；selection有平台但audit失败=`source_protection_not_audited_transfer_stop`；audit通过=`source_protection_evaluation_candidate_requires_independent_confirmation`。

## Risks / Trade-offs

- **evaluation曾参与checkpoint选择** → 结果只用于方法开发，不作为论文确认。
- **固定网格可能错过窄最优值** → 窄峰不视为稳健证据，不扩网格。
- **标准化改变绝对logit尺度** → beta0保持严格单调等价，非零beta具有跨用户一致单位。
- **一个缓存run计算量约三次beam** → 冻结一次运行，不为每个beta重复GPU推理。

## Migration Plan

新增配置与launcher显式启用，不改变已有union模型和checkpoint。缓存完成后由writer自动生成本地分析JSON；回滚只需停用新实验入口。

## Open Questions

无；split、beta、门槛与预算均冻结。
