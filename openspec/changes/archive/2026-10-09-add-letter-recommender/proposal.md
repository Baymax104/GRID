## Why

LETTER-TIGER 的全词表 T5、EOS 与温度训练目标无法由现有 TigerDecoder 忠实表达。

## What Changes
- 新增独立 LETTER T5 backbone、词表映射、约束 beam 与温度 CE。
- 本提案不含数据/Lightning/pipeline，也不复用已有推荐模型或指标实现。

## Capabilities
### New Capabilities
- `letter-recommender`: 作者 T5/token/温度/目录约束算法。
### Modified Capabilities
无。

## Impact
新增 src/recommendation/letter 算法及内存测试；使用现有第三方 Transformers。
