# sasrec-backbone Specification

## Purpose
定义 SASRec 官方算法骨干的商品与位置 embedding、因果注意力及残差计算，规定逐位置正负 BCE、正则化和共享目录打分，通过公式与梯度核验保证 padding 和因果边界正确。
## Requirements
### Requirement: Official SASRec forward computation

Backbone SHALL 使用作者固定 commit 的 item embedding 缩放、槽位位置 embedding、Q-only LayerNorm、无 output projection 的因果多头 attention、官方 residual/FFN 顺序和末端 LayerNorm。padding 商品 SHALL 恒为零输入，block 后清零，LayerNorm epsilon SHALL 为 1e-8。

#### Scenario: Fixed weights match official equations
- **WHEN** 在 dropout 关闭情况下对含左 padding 的输入使用固定权重
- **THEN** 单头与多头前向 SHALL 与独立官方公式参考在数值容差内一致

#### Scenario: Future actions cannot change earlier states
- **WHEN** 修改某个有效商品之后的未来商品
- **THEN** 更早位置的表示 SHALL 不变

### Requirement: Official pointwise objective and tied scoring

系统 SHALL 共享输入/输出商品 embedding，对所有非零正标签位置计算官方带 1e-24 的正负 BCE 和 item/position embedding L2，按有效位置平均。padding SHALL 不贡献 BCE。最后槽位 SHALL 与同一 embedding 表点积评分。

#### Scenario: Loss and gradients
- **WHEN** 输入有效正负标签和 padding 标签
- **THEN** loss SHALL 与官方公式一致，梯度 SHALL 到达 embedding 与 Q/K/V；padding 行 SHALL 不通过有效 embedding 产生梯度

#### Scenario: Invalid objective is rejected
- **WHEN** 全部正标签为 padding 或输入 ID 越界
- **THEN** 系统 SHALL 明确拒绝输入
