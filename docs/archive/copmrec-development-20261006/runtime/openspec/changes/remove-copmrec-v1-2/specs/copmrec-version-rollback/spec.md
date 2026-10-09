## ADDED Requirements

### Requirement: Remove v1.2 runtime support
系统 SHALL 删除v1.2组件、模型与experiment配置、根训练/推理入口和专属测试，活动代码不再提供v1.2。

#### Scenario: Runtime entrypoint removal
- **WHEN** 用户要求删除v1.2退回v1.1
- **THEN** 本地及远端受管代码中v1.2的六个运行文件不再存在，活动代码没有对应引用

### Requirement: Restore calibrated v1.1
系统 SHALL 使用v1.1四路head、批量评分及校准排序权重0.05726763550972437，并保持既有训练、恢复和评价契约。

#### Scenario: Existing v1.1 launch
- **WHEN** 使用copmrec_v1_1_train或inference入口
- **THEN** 使用BatchedRelevanceCoPMRec和head输入[h,v,h*v,d]，配置、脚本及checkpoint回归通过

### Requirement: Preserve historical evidence
系统 SHALL 保留历史核验与实验重资产，并在研究状态中明确v1.2已撤回而v1.1为当前版本。

#### Scenario: Withdrawn implementation evidence
- **WHEN** 清理v1.2代码
- **THEN** 历史文档标记撤回，原实现规格归档，W&B结果及checkpoint保持，撤回不被宣称为效果否证
