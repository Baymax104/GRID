## Why

保留v5从头50k的NDCG增益，但Recall未达双8%；v5.1双目录干预已反驳并停止。实际cold错误曝光及训练CE支持集与mixture／部署的差异支持一个最小、有界的训练目标鉴别，不证明cold是单一根因。

## What Changes

- 新增v5派生v5.2，只让训练content CE包含完整目录cold logits，保留整个随机起点50k协议及原单目录模型。
- 固定dense_ce_support模型／CP／metadata契约、薄配置和根脚本，新增聚焦验证，旧运行文件字节保持。
- 占用原150k内最后1train／50k与最后1完整单卡Val，真实来源、预算、checkpoint与完整输出审计后判断原双8%；Test0，不自动扩大预算。

## Capabilities

### New Capabilities

- `copmrec-unified-full-catalog-ce-50k`: 完整目录content CE的随机初始化连续50k鉴别及来源评价边界。

### Modified Capabilities

无。

## Impact

新recommendation类、model／experiment配置、根训练／推理脚本与聚焦测试；研究状态与累计账本、只读编排／审计记录。无新依赖，不改baseline默认行为、不增加query／参数，不使用推荐checkpoint初始化。来源和停止决定见`docs/copmrec-unified-full-catalog-ce-50k-research.md`。
