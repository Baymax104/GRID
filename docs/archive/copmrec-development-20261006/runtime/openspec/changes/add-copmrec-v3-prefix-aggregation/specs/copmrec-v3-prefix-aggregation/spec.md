## ADDED Requirements

### Requirement: 固定逐层内容聚合

v3 SHALL 在首层SID搜索使用合法子前缀后代内容最大值、后续层使用后代logsumexp，并在各层合法兄弟节点内归一化，与合法生成条件概率通过原全局alpha混合。

#### Scenario: 相同父前缀的条件分布
- **WHEN** 枚举商品计算首层与后续层内容条件概率
- **THEN** v3结果与首层max、后续mass的独立枚举一致，非法分支为负无穷，合法条件概率和为1

### Requirement: 训练与推理使用相同聚合

v3 SHALL 以相同逐层聚合计算teacher forcing混合NLL与beam搜索分数，保持SID CE、content CE和混合NLL三项等权及全部推荐参数可训练。

#### Scenario: 逐步解码与teacher forcing
- **WHEN** 在eval模式使用相同输入逐步计算目标路径概率
- **THEN** 其平均负对数概率与训练混合项一致，梯度对encoder/decoder/content投影和alpha有限，初始化和参数数量与v0一致

### Requirement: 独立版本与恢复契约

v3 SHALL 在checkpoint与candidate/path trace记录v3身份及逐层聚合，恢复时校验版本和固定聚合/目标契约。

#### Scenario: 错误版本或缺少聚合契约
- **WHEN** 用v0 checkpoint或缺少/改变v3聚合契约的checkpoint恢复v3
- **THEN** 模型明确拒绝，合法v3恢复产生相同预测，标签变化不改变预测

### Requirement: 匹配真实v0运行协议

v3 SHALL 直接继承真实v0的训练/推理数据、optimizer、scheduler、trainer、dense验证选点、content终排及beam20，使用独立experiment与根脚本。

#### Scenario: 双卡训练入口
- **WHEN** 脚本收到NPROC_PER_NODE=2、devices=[0,1]、notes与用户override
- **THEN** 调用uv run torchrun和统一src.main入口，默认50k更新、每卡batch128、累积1、500验证间隔；末尾override优先，默认不dry-run

#### Scenario: 旧版本保持
- **WHEN** compose v0/v1/v1.1/v2或执行旧processor测试
- **THEN** 有效行为配置及原mass/max算法保持一致
