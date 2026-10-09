# letter-recommender Specification

## Purpose
定义 LETTER 独立 T5 推荐骨干、作者 token 映射、温度损失和共享权重，约束基于唯一 SID 目录的 Trie 生成，保证候选可逆、合法且无重复，并保留历史商品与冷启动目标。
## Requirements
### Requirement: Independent T5 and temperature loss
系统 SHALL 使用独立标准T5和作者token映射、EOS，训练loss SHALL 为CE(logits/tau)，tau SHALL 有限且正，推理 SHALL 使用原logits。
#### Scenario: Temperature and tied embeddings
- **WHEN** 固定输入与模型权重，比较tau=1及非1
- **THEN** loss SHALL 与独立CE一致，输入/输出embedding SHALL 共享权重

### Requirement: Full catalog constrained generation
系统 SHALL 以唯一合法SID目录构建Trie并输出唯一商品keys，不过滤历史商品或cold目标。
#### Scenario: Constrained tiny catalog
- **WHEN** 使用微型随机T5执行beam
- **THEN** 全部输出 SHALL 属于目录、每用户无重复，SID和原key可逆
