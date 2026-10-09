# letter-unique-sid-export Specification

## Purpose
规定 LETTER 完整目录四码 SID 导出的有界碰撞修复和末层唯一分配，保持前三层码与训练行为，容量不足时明确失败，并登记 GRID 适配及真实目录合法性、确定性核验。
## Requirements
### Requirement: 容量内唯一四码导出
系统 SHALL 保留作者有界修复；当仍碰撞且每个前三层prefix人口不超过末层容量时，按该prefix全部商品做末层一对一最小总残差距离分配。

#### Scenario: 相同内容和已占用末码
- **WHEN** 两商品内容相同且同prefix还有未碰撞商品
- **THEN** 导出包含所有商品、保持前三层码、全部末码唯一，不能撞到未碰撞邻居。

### Requirement: 明确容量失败
系统 SHALL 在prefix人口超过末层容量时拒绝导出，不添加第五位或改变前层码。

#### Scenario: 四个末码服务五个同prefix商品
- **WHEN** 一个prefix有五个商品而末层只有四个码
- **THEN** 报告容量不足，不返回碰撞bundle。

### Requirement: 适配登记和复核
系统 SHALL 登记硬匹配为GRID适配，验证训练行为不变、真实目录合法唯一及确定性。

#### Scenario: 重复导出
- **WHEN** 同checkpoint与同排序输入在相同环境重复导出
- **THEN** 结果相同且全目录唯一。
