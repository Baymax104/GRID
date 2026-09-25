## Context

现有`liger_source_protection_v1`缓存保存mass20、max20、mass30、union池及内容分数，足以离线精确构造来源约束Top10。此前beta方案对所有mass候选连续加分，不能直接限制max-only候选占据多个Top10位置。

## Decisions

1. `q=1`固定为最小非零准入，不扫描其他quota。
2. max-only定义为出现在max20、未出现在mass20、且不属于全局cold集合；cold候选是所有控制臂共有候选，不计入max增量槽。
3. 先屏蔽除内容分数最高者之外的max-only候选，再按原始内容分数稳定排序Top10；没有合格max-only时退化为mass20+cold内容排序。
4. selection门槛为相对mass20和mass30均满足NDCG@10点估计大于0、Recall@10点估计不降。union仅作描述性比较。
5. selection未通过时返回`single_max_slot_selection_gate_failed_stop`并保持audit为空；通过后，audit要求相对mass20和mass30的NDCG@10配对95%CI下界均大于0且Recall@10点估计均不降。
6. audit失败返回`single_max_slot_audit_failed_stop`；通过返回`single_max_slot_evaluation_candidate_requires_independent_confirmation`。

## Risks

- `q=1`可能错过其他quota，但扫描会把一次可证伪验证变成调参，因此不扩展。
- 设计受到selection来源冲突统计启发；audit半区仍未查看，可用于一次冻结检验，但通过仍需独立seed或数据集确认。
- 若失败，停止当前实例的候选转化探索，不以更换quota或公式续跑。
