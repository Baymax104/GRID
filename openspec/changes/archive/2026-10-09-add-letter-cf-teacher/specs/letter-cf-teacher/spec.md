## ADDED Requirements

### Requirement: 独立协同向量训练
系统 SHALL 使用独立LETTER域模型训练32维商品表，不导入项目其他模型；只有training交互参与梯度与负样本排除集合。

#### Scenario: 原始商品0及截断历史
- **WHEN** training行包含原始商品0及超过history50的历史
- **THEN** 商品0映射非零模型ID，负样本排除完整行，逐有效位置训练下一商品。

### Requirement: 统一入口与身份
系统 SHALL 通过src.main/Hydra/root scripts训练及导出，使用evaluation NDCG10选优且拒绝错目录checkpoint。

#### Scenario: 错目录恢复
- **WHEN** 同shape不同商品目录恢复checkpoint
- **THEN** 在消费模型前拒绝恢复。

### Requirement: 导出可消费CF
系统 SHALL 使用共享writer导出全部目录的keys和32维有限商品向量，显式checkpoint来源且单进程执行。

#### Scenario: 对齐tokenizer
- **WHEN** 导出CF与对应内容bundle交给LETTER tokenizer
- **THEN** 全目录key一致且形状为items乘32；cold商品仍保留。

### Requirement: 交付验证
系统 SHALL 验证公式、负采样、配置/脚本与真实输入有限dry-run，并交付九槽位完整阶段命令；不自动启动正式实验。

#### Scenario: 人工启动Testing
- **WHEN** 用户复制Testing命令但未填写本单元validation best
- **THEN** shell守卫先报缺少来源，不启动推理。
