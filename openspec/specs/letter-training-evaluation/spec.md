# letter-training-evaluation Specification

## Purpose
定义 LETTER 独立模块的训练、验证、checkpoint 身份恢复和推理评价，保证全目录合法唯一生成与用户级 Recall/NDCG 计算，并约束分布式指标按用户总数正确归并。
## Requirements
### Requirement: 独立训练与身份恢复
系统 SHALL 通过独立LETTER模块训练、验证、恢复checkpoint及生成，不调用其他模型。
#### Scenario: checkpoint目录被替换
- **WHEN** 恢复输入目录或训练模型配置与checkpoint身份不同
- **THEN** 系统拒绝恢复。

### Requirement: 共同全目录评价
系统 SHALL 在全目录生成合法唯一商品，不过滤历史，并按用户计算Recall/NDCG@5/@10。
#### Scenario: 分布式验证
- **WHEN** 不同rank处理不同用户
- **THEN** 指标按用户总数归并，不平均rank均值。
