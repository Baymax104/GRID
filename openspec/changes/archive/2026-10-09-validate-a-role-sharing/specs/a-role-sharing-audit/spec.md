## ADDED Requirements

### Requirement: 冻结角色分解
系统 SHALL 严格加载原A并核验零干预与路径梯度恒等，禁止优化器更新，异常时清除hook。

#### Scenario: 路径分解成功
- **WHEN** 对最小合法目录运行support审计
- **THEN** 两个角色梯度之和与共享梯度在容差内一致，模型状态不变

### Requirement: 独立标签与匹配扰动
系统 SHALL 仅从training构造方向，evaluation用户与support用户互斥；各非零方向联合范数匹配，evaluation标签不得改变条件预测。

#### Scenario: 更换评价标签
- **WHEN** 输入和support不变且更换合法评价目标
- **THEN** 各条件TopK与扰动方向不变，仅目标统计改变

### Requirement: 可审计有界证据
系统 SHALL 通过共享writer导出用户级目标概率、精确rank、TopK、support梯度统计、指纹及条件定义，并核验概率归一化。

#### Scenario: 无模型实验的验收
- **WHEN** 运行单元测试、Hydra compose和脚本stub
- **THEN** 不加载外部数据或启动真实Trainer，完整实验仍需手动运行
