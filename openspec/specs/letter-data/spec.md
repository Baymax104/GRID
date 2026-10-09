# letter-data Specification

## Purpose
定义 LETTER 按原始商品 key 对齐内容、32维协同向量和唯一四码目录的输入契约，保持共同数据划分及逐前缀监督、历史截断和 EOS，并明确拒绝缺失或重复目录数据。
## Requirements
### Requirement: 独立且严格的LETTER数据契约
系统 SHALL 按原始key对齐独立LETTER目录、内容及32维CF，并保持共同split。
#### Scenario: 逐前缀监督
- **WHEN** 训练记录包含至少两个商品
- **THEN** 系统生成每个非空前缀的下一商品目标，历史截断到配置长度并追加EOS。
#### Scenario: 缺失或重复目录
- **WHEN** SID重复或CF缺少目录商品
- **THEN** 系统拒绝继续而不是按行号对齐或追加碰撞码。
