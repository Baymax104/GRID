# liger-hybrid-retrieval Specification

## Purpose
定义 LIGER 生成候选、冷启动补入及内容重排的联合检索流程，规定无效候选处理、完整目录输出和三种评价模式，确保不同路径的候选身份、评分语义及最终商品排名一致可核验。
## Requirements
### Requirement: 生成与重排
模型 MUST 默认生成20个候选，解码全词表 SID，将合法商品与明确的冷启动集合并集去重后按共享内容投影 cosine 排序；不得引入 CGBS 混合概率。

#### Scenario: 生成与重排验证
- **WHEN** 冷启动商品未进入生成候选
- **THEN** 它仍能由补入通道进入最终 TopK

### Requirement: 无效候选与输出
非法生成 MUST 丢弃，不能用 argmax 错映射到首商品；不足 TopK 用 -1 SID 补位。输出 MUST 为 keyed ModelOutput。

#### Scenario: 无效候选与输出验证
- **WHEN** 非法或重复生成和不足候选
- **THEN** 输出不含伪造商品、重复商品，padding不命中

### Requirement: 评价模式
模型 MUST 支持 dense、generative、hybrid 评价；默认训练验证 dense、独立推理 hybrid，允许显式 override。

#### Scenario: 评价模式验证
- **WHEN** 切换评价模式
- **THEN** 输出保持共同 SID 指标协议并可严格恢复 checkpoint
