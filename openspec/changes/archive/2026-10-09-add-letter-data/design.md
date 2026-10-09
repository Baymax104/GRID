## Context
统一数据包含原始商品 key 的 sequence_data。LETTER 使用四个学习码及 EOS；CF 是显式上游输入，不能使用现有推荐模型生成。
## Goals / Non-Goals
支持 keyed 内容/CF 对齐、逐前缀训练和共同 split；本变更不实现 CF teacher 或修改其他模型。
## Decisions
目录 token 编号按作者字符串排序。训练样本为每个非空历史前缀，历史最多20个商品；验证/测试只消费最后目标。输入尾部加入EOS，动态右padding。
## Risks / Trade-offs
内容与CF必须覆盖同一目录且CF宽度32；不从张量行号推断key。重复SID拒绝输入。
