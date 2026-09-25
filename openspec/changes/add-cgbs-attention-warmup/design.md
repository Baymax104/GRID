## Context
验证content CE后期上升；表达能力不足未成立，用户授权一次组合验证。
## Goals / Non-Goals
增加历史隐状态的可学习加权读取，保持Transformer、目录和评分规则。不是新增独立Encoder或动态混合门控。
## Decisions
对每个Encoder有效位置打分s_t=w^T h_t，掩码softmax得权重，加权和后送入128→512→128 GELU MLP并归一化。无bias（softmax平移不变）；w零初始化，使初始pooling等于mean且不扰动backbone随机流。pooling包含现有mask下的有效SID及分隔符，与原mean的可见位置一致；屏蔽padding，全空mask拒绝。

query_pooling默认mean，attention才加入参数和catalog_contract.query_pooling；历史checkpoint保持兼容，attention/mean混用被拒绝。训练和推理复用原_query，无目标label输入。

沿用warmup子类，10000步content CE仅更新Encoder/shared SID/attention/MLP，之后20000步全模型联合，不重置Adam。最佳checkpoint在step>10000选择。保留val/content_loss和阶段日志。独立配置不覆盖5k+35k旧协议。
## Risks / Trade-offs
增大容量不保证泛化，已知512负面结果保留。总30k、Decoder专属20k，不可与40k比较后声称纯attention效果。若无验证推荐增益，停止组合，不自动扫描宽度/attention头数/阶段比例。完整实验由用户手动启动。
