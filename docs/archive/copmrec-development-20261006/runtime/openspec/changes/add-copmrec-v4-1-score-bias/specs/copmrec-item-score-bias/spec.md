## ADDED Requirements

### Requirement: 独立seen商品打分截距

系统 SHALL 在v4 cosine/temperature目录分数后加入零初始化seen商品scalar bias，使用同一logits服务content CE、mixture NLL及dense排序；bias MUST 不回写历史表示，cold项及其梯度为0。

#### Scenario: 零初始化保持基础行为
- **WHEN** 相同seed及v4权重初始化v4.1且bias全零
- **THEN** RNG、历史query、三loss与dense输出逐数值相同，cold非有限bias也被mask为0

#### Scenario: 损失的直接监督
- **WHEN** 分别对三项loss反向
- **THEN** content及mixture对seen bias有有限非零梯度，cold梯度为0，SID loss对bias无直接梯度

### Requirement: 显式v4 weights-only初始化和严格恢复

系统 MUST 验证原v4 checkpoint版本、learned alpha、catalog/完整参数/来源/残差训练协议，只读取其权重并新增零bias；v4.1恢复 SHALL 核对独立bias与warmstart来源契约。

#### Scenario: 初始化后保持v4行为
- **WHEN** 使用已验证v4 checkpoint及显式pretrained_v4_checkpoint
- **THEN** 原权重严格保留、bias全零、新optimizer状态为空，不恢复原global_step或scheduler

#### Scenario: 错误来源或恢复配置
- **WHEN** 传入非v4、fixed-alpha、catalog不一致、缺参数、bias scale或来源不匹配checkpoint
- **THEN** 初始化或恢复失败，不静默兼容

### Requirement: 匹配zero-bias与单卡dense契约

系统 SHALL 支持scale1学习bias与scale0冻结bias；两臂共用既有训练协议和两组optimizer，新增bias采用item组LR。v4.1 MUST 仅允许dense/content部署，独立推理进程数为1。

#### Scenario: zero-bias对照
- **WHEN** score_bias_scale=0
- **THEN** bias不进入optimizer，原残差仍更新，base/item组LR与学习臂相同

#### Scenario: 统一入口及参数透传
- **WHEN** 根训练脚本收到notes/dry-run/额外override或单卡推理命令
- **THEN** 经src.main/Hydra执行，quoted引用与用户override保留；多进程独立推理被拒绝

### Requirement: 有界研究与真实性

研究记录 MUST 关闭原3×6000阶段，再登记新score-bias问题最多2×6000步，wholethread总5臂/30000步；晋级依据匹配推荐净收益，不能用曝光减少、训练loss或参数范数替代。

#### Scenario: 负向或证据不足
- **WHEN** 新bias没有匹配control之外的推荐增量，或只减曝光而损害指标
- **THEN** 关闭bias机制迭代、停止bias/temperature/LR序列，不重置预算或降低10%目标；整体效果与组件归因分别判断，若有臂达到同split LIGER两项10%门槛，仍可按Validation NDCG10选择唯一winner进行Testing，但不得把不确定的bias增量称为已证实
