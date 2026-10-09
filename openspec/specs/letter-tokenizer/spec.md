# letter-tokenizer Specification

## Purpose
定义 LETTER 独立量化器的 MLP、残差 VQ、STE、协同与多样性目标，规定 constrained K-means 初始化、分组状态恢复和纯编码边界，并通过有界修复检查完整目录 SID 碰撞。
## Requirements
### Requirement: Independent official tokenizer
系统 SHALL 独立实现作者 MLP、逐层残差 VQ、STE、均值 VQ loss、batch CF CE 与同组正例 diversity，不依赖已有领域模型。

#### Scenario: Fixed weights and gradients
- **WHEN** 输入固定权重和正例索引
- **THEN** SID、重建、各项 loss 与梯度 SHALL 匹配独立作者公式参考，CF 不直接更新 codebook

### Requirement: Persistent clustering and bounded export
系统 SHALL 保存初始化/分组状态，使用 constrained K-means，纯编码不抽正例；SID 导出 SHALL 全目录修复碰撞且残余碰撞失败。

#### Scenario: Resume and collision
- **WHEN** 恢复 state_dict 并编码相同输入，或输入无法唯一量化的商品
- **THEN** 恢复输出 SHALL 一致，碰撞修复 SHALL 有界结束并明确失败
