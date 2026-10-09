## ADDED Requirements

### Requirement: Original v0 configuration identity
系统 SHALL 将BMX-116已有基础CoPMRec定义为v0，版本化入口保留原始训练与推理配置行为。

#### Scenario: Original training protocol
- **WHEN** 选择copmrec_v0_train
- **THEN** 使用FileDataModule全evaluation数据、dense验证、500微批验证间隔、原始主干/optimizer/scheduler与50k更新预算，不使用selection/audit切分

#### Scenario: Original inference protocol
- **WHEN** 选择copmrec_v0_inference
- **THEN** 使用FileDataModule读取testing并以hybrid预测，training_model_config为null，保持原产物协议

### Requirement: Explicit closed version isolation
系统 SHALL 显式保持v1/v1.1已结束迭代时的可执行行为，不因v0配置恢复隐式改变其数据和trainer。

#### Scenario: Resolved configuration comparison
- **WHEN** 比对恢复前后v1/v1.1训练与推理装配
- **THEN** 数据、模型、回调与trainer的resolved行为相同，仍为selection/audit与2500训练验证间隔

### Requirement: Auditable configuration verification
系统 SHALL 保存原v0实际配置与版本化更改的验证依据，不把实现核验视为正式推荐效果。

#### Scenario: Delivery
- **WHEN** 交付统一配置
- **THEN** 聚焦配置/脚本检查和OpenSpec strict通过、node1指纹一致、Mutagen三个session无conflict，不启动或停止完整训练

### Requirement: Unified v2 evaluation data
系统 SHALL 按用户确认使v2与真实v0使用同一evaluation/testing数据链路及基础训练配置，保留v2融合终排与其分布式安全设置。

#### Scenario: Full evaluation and testing
- **WHEN** 运行v2训练或推理
- **THEN** FileDataModule在训练读取全evaluation、预测读取testing，不使用selection/audit，500步验证，以融合hybrid分数验证/选点/预测，lambda0.01和beta0.5保持
