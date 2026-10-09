## Context

LETTER 需要与共同内容目录逐key对齐的32维 SASRec 商品表。已有推荐基线为50维且用户明确禁止复用项目模型。作者未公布teacher训练细节，此实现是明确登记的GRID上游适配。

## Goals / Non-Goals

目标：独立SASRec公式、严格training梯度边界、统一入口、可验证的CF导出与完整命令。非目标：启动50k正式实验、声称复现作者teacher的未知训练设置。

## Decisions

- 依据kang205/SASRec commit e3738967fddab206d6eeb4fda433e7a7034dd8b1：Q=LN(x)，K/V=x，attention无output projection，query residual、逐点FFN与BCE。
- 固定hidden32、history50、2blocks/1head/dropout0.5，Adam lr0.001/betas0.9,0.98，无scheduler，单GPU batch128，50k steps，每1000步evaluation，NDCG10选优。该训练预算是GRID适配，不增加搜索预算。
- 训练只读取training，负采样排除完整training行；评价只选checkpoint，不进入梯度或负采样。testing不用于teacher。
- 内容bundle只读取keys作为共同目录；原始商品0映射模型非零ID，模型0是padding。CF导出原始keys及未归一化商品表，含全部目录商品；cold商品无正交互监督，可能收到负采样梯度，不声称具备充分协同信号。
- own data adapters与Lightning模块；复用公共SequenceDataset/FileDataModule、reader、MetricEngine、artifact resolver/writer。导出单进程避免分布式重复目录。
- checkpoint校验目录hash及结构协议；所有可恢复属性保持普通Python类型。

## Risks / Trade-offs

未知作者teacher设置和随机初始化差异不能宣称精确复现。CF/table已具备训练链路不等于有效效果；正式训练及全矩阵由用户手动开始。

## Migration Plan

向前新增入口，旧LETTER外部CF输入仍兼容。新命令优先使用本teacher明确记录来源的产物。

## Open Questions

正式表现待用户训练；不阻塞可运行准备，不用合成指标替代真实结果。
