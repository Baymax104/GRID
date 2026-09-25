## ADDED Requirements
### Requirement: 同池排序配对
系统 SHALL 以dense候选、原生成有效候选、cold并集作为两臂同一候选池，纯内容排序复现全目录dense TopK。
#### Scenario: 联合排序预测
- **WHEN** 用户启用paired_rerank且提供标签用于记录
- **THEN** 输出同池dense和joint排名及可重算证据，不使用标签构建池或评分
### Requirement: 一致生成评分
系统 SHALL 对池内所有商品按完整SID的完整词表条件log概率求和评分，与是否被beam生成无关。
#### Scenario: 商品未被beam选中
- **WHEN** 商品来自dense或cold候选
- **THEN** 它仍获得相同teacher forcing评分，不以缺席生成列表代替分数
#### Scenario: 分块或扩beam
- **WHEN** 修改评分分块大小或HF生成扩展encoder容器
- **THEN** 用户与encoder对应不变，评分和最终排名保持数值一致
### Requirement: 汇总与兼容
系统 SHALL 复用共享writer输出配对指标、得失命中与采样区间，拒绝无效证据并保持原模式兼容。
#### Scenario: 原模式
- **WHEN** 未启用paired_rerank
- **THEN** 默认行为和旧checkpoint保持兼容
