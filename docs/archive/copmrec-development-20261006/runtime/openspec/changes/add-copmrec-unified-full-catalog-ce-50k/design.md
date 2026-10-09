## Context

v5已证明从step0共同训练可有NDCG收益，R10+3.60%未确认；v5.1对v5双指标CI为负，已停止。只读三输出证据显示cold曝光216／6644／11113，v5已有seen命中前209个cold槽，但丢失关联弱且无Top11。当前training CE cold=-100、mixture full logits、部署cold eligible可直接核验，支持一次训练支持集干预，不支持因果归因或cold完全无监督。

## Goals / Non-Goals

目标：v5派生单模型随机初始化连续50k、完整目录CE支持；在原预算最后一次完整Val验证整体双8与覆盖／排名取舍。保持原同seed分母、CI、来源和成对复现要求。

非目标：权重／温度／alpha／CP扫描，cold硬排除或bias，额外query／teacher／pool，自动seed43或Testing，回填历史源码缺口。

## Decisions

新`UnifiedFullCatalogCECoPMRec`固定版本v5.2，父契约增加5字段`dense_ce_support`。只改training content CE支持集；SID／mixture、单query／单目录、history／cold残差／rank／选择及optimizer日程沿用v5。保留原decoder→目录projection的调用与RNG顺序，优先新增文件不改旧396字节。CP契约严格拒绝跨方法／错support与weights-only warm-start。

正式1train50k＋1singleGPUfullVal计入原150k账本；actual100k已完成，最后槽预先指定但未启动不记完成。主要门禁native双8＋双CI正，v5增量与cold占位辅助不能替代它；也不任意添加v5非劣门槛否决实际达到用户主要目标的模型。辅助预测失败须收缩主张；无明确推荐收益停止，不自动追加模块或预算。

## Risks / Trade-offs

新增cold负例竞争可能损害51个cold目标，需全量保留；梯度会改变query与projection，不能固定参数归因。实际效果与计算成本待运行，零新参数不等于同FLOPs。剩1槽无法完成两模型seed43复现，整体goal不得提前完成。

## Validation

先聚焦CE cold梯度、all-seen/eval等价、loss调用顺序、cold残差、完整CP契约和原核心回归；薄配置compose／脚本quoting、notes、dry-run、override／singleGPU guard。实际CPU和DDP2一步smoke、官方同步后唯一正式运行。独立训练50k／100raw点／ownbest／state／source审计，再singleGPUstrict restore及完整原始用户指标／paired／case审核，核成本与停止决定，OpenSpec strict。
