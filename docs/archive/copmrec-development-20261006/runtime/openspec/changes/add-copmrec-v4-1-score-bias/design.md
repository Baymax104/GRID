## Context

原v4阶段已关闭，当前开发checkpoint为nj9elah1 Validation-selected best6000，来源SHA7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055。其源码和全量Evaluation已独立核验；Testing收益不能用新Validation替代。新机制依据当前两组稳定的商品误排，而非未达到10%这一事实。

## Goals / Non-Goals

**Goals:** 增加独立seen商品logit截距，保留共享内容/协同历史表示与v0三loss；严格初始化及匹配zero-bias续训；单卡dense部署和可复算来源。

**Non-Goals:** 不恢复mixed终排、候选路线、temperature/alpha或LR扫描；不拟合Validation频率，不宣称cosine表达不可能或概率校准已证实。

## Decisions

1. 新`ItemScoreBiasCoPMRec`派生自v4，版本`v4.1`。`dense_logits(q)=v4.dense_logits(q)+seen*b`，加在temperature之后；b为12101个零初始化参数，零初始化不消耗RNG，cold masked_fill为0。`score_bias_scale`仅0或1；0冻结并绕过加法。b不进入历史encoder。
2. 三loss保持各1；content CE及合法Mass-prefix mixture NLL读取同一目录logits。SID CE无b直接梯度。此次v4.1只允许dense evaluation/prediction、content final scoring、关闭candidate/path trace，避免扩展无关trace协议。
3. 显式`pretrained_v4_checkpoint`读取已声明`pretrained_checkpoint_path`，父级v0初始化配置设null。验证原v4版本/learned alpha/源hash/catalog/完整旧参数集合/residual contract后weights-only严格加载；新增b归零，新optimizer/scheduler/global_step。原v4来源version必须独立记录，不能把桥接校验副本当作原文件元信息。
4. v4.1正常checkpoint恢复严格核对score-bias contract、scale、cold策略、来源，以及原v4残差/alpha训练契约；v4 checkpoint只能经显式weights-only初始化入口进入。保存v0→v4→v4.1来源链，实际上游artifact由公共resolver与lineage callback记录。
5. b与已有collaborative residual共用item参数组，固定LR0.002/weight_decay0.035；主干组LR0.0001，总两组不变。不给b新增独立LR超参。matched control冻结b、残差仍训练，同数据/RNG起点/6000步/warmup300/globalbatch256/FP32/seed42/1000步验证。gradient clipping及基础参数会随b梯度共同变化，不声称仅最终打分值的纯因果效应。
6. 两臂均从同一selected best6000出发，按dense evaluation NDCG10选best，同时预先固定step6000标量比较。每臂一次单卡完整Evaluation保存真实输出、配对指标及固定offender损益；只在验证选择的结果满足原两项10%门槛后消耗一次全线程剩余Testing确认。

Validation门槛相对同split固定LIGER5azn5vm0：R10≥.09877029021151008（2209命中）、N10≥.0511173919307933。Testing验收独立相对042139al：R10≥.07855386128873586（1757命中）、N10≥.04002978116676896，绝对阈值不可跨split复用。

若两臂均符合Validation门槛且NDCG10精确相同，选择结构更简单的冻结bias对照；此tie-break在control启动及完整Validation结果产生前登记。

## Risks / Trade-offs

- 商品截距可能强化常见商品或损害cold排名 → cold参数严格0、warm/cold及整体指标如实报告；零bias不保证cold相对排名不变。
- 重复错误曝光不是概率校准证据 → 以匹配后的推荐净收益作为主判据，曝光仅机制描述。
- extra6000步、best选择与clip耦合 → 匹配zero-bias续训、同step6000检查与完整Validation配对，保留单seed/重复开发选择边界。
- 初始化误用v0/旧optimizer → 根脚本ckpt_path=null、显式v4 source与严格结构/来源测试；真实byte检查和双卡production-batch dry-run先于正式任务。

## Migration Plan

新增component/experiment及薄脚本，v0/v4入口默认不变。实现及检查通过后Mutagen flush与源hash核验，执行一次双卡装配dry-run，再串行两臂正式训练。整体效果与bias归因分开判断：达到同split LIGER两项10%门槛的臂按Validation NDCG10选择唯一winner进入Testing；即使winner整体达标，bias对control增量不确定也不声称bias有效。负向或增量证据不足关闭bias机制迭代，不追加bias/temperature/LR扫描；两臂均未达整体门槛则保留原v4。此规则在两个新训练启动前固定。

## Open Questions

该独立截距能否在匹配续训之外改善推荐；稳定offender损益是否同时改善。由唯一一对matched臂回答，不把未知作为后续自动扩预算理由。
