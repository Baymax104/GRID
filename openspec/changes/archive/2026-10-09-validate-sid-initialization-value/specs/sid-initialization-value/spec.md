## ADDED Requirements

### Requirement: 五seed冻结testing入口

系统 SHALL 提供Beauty seed42至46的A/深层残差冻结推理入口，显式引用best-val checkpoint及原上游产物，使用testing、beam10、单GPU和同用户prefix trace；SHALL 支持单seed及单条件选择、dry-run、notes和额外override最后优先。

#### Scenario: 全部冻结模型
- **WHEN** 用户运行默认testing包装脚本并指定data-dir
- **THEN** 依序运行恰好十项既有checkpoint推理，使用统一src.main入口，不启动训练，任一失败即停止

#### Scenario: 独立两卡并行
- **WHEN** 用户分别指定full_content/gpu0和deep_residual/gpu1及不同端口
- **THEN** 每组五项单卡推理，逻辑devices=[0]，任务名包含训练seed与条件，配置记录原训练run及预期digest

#### Scenario: testing判读边界
- **WHEN** 结果参与正式汇总
- **THEN** 核对实际lineage、冻结配置及用户标签对齐，全部seed报告；不得据testing重新选择checkpoint或修改方法

### Requirement: 深层方向保持的范数校准

系统 SHALL 提供deep_norm_calibrated初始化，固定实验eta=0.5，仅改变深层语义均值的径向强度并重新匹配全层population std，不新增训练参数。系统 MUST 保留默认A行为、首层、去重层、未用码和随机流，并以独立checkpoint契约记录校准参数。

#### Scenario: 恒等边界
- **WHEN** 校准指数为1
- **THEN** 初始化权重与原A逐位一致，历史A契约仍仅用于full_content模式

#### Scenario: 安全径向变换
- **WHEN** 指数为0.5
- **THEN** 非零方向保持，范数大于等于1e-8时范数比按平方根压缩，零向量保持零，近零范数使用1e-8截断，无非有限数值

#### Scenario: 有限手动筛选
- **WHEN** 用户指定校准条件与seed45或46
- **THEN** 仅启动该seed一个两卡20k训练，配置记录对应原A run及校准协议，支持既有notes/dry-run/override约定

### Requirement: 固定实现跨seed复核

系统 SHALL 支持paired条件按同seed依次训练full_content和deep_residual，单独full_content可用于补跑；原both含义保持不变。seed43、44的成对训练 MUST 保持A实现、PCA/码本来源、bank_seed42、20k预算、batch、优化器及checkpoint规则不变，记录结构化复核标识。

#### Scenario: 四次手动训练
- **WHEN** 用户分别执行seed43和44的paired命令
- **THEN** 每命令依次启动A和残差两个新训练，每个两个GPU进程，失败即停，不恢复checkpoint或运行其他消融

### Requirement: 冻结模型推理验证

系统 SHALL 提供first_only和deep_residual的推理组件配置，继承训练初始化契约及标准推理行为，通过既有根推理脚本和src.main运行；使用明确最佳checkpoint、evaluation、beam10、用户key与prefix trace。完整推理 MUST 由用户手动启动。

#### Scenario: 训练checkpoint加载
- **WHEN** 同一条件由训练配置切换到推理配置
- **THEN** catalog/初始化契约和state_dict结构兼容，trace启用，metrics=null，不启动训练或改变checkpoint权重

### Requirement: 分层初始化条件

系统 SHALL 支持 A 默认完整内容、仅第一层和深层残差三种初始化口径，保持去重码、未用码、backbone、损失、随机流及参数数量一致。

#### Scenario: 仅第一层
- **WHEN** A 使用 first_only
- **THEN** 首层与A逐位一致，其余层与相同seed的随机基准逐位一致

#### Scenario: 深层残差
- **WHEN** A 使用 deep_residual
- **THEN** 首层与A一致，后续语义层按设计中固定PCA、坐标均值及总中心化能量匹配方案初始化，无逐行L2或在线内容分支

### Requirement: 可靠码本与加载契约

系统 MUST 使用公共artifact解析与lineage记录量化器输入，检查SHA256、结构、维度和既有SID抽样身份，记录初始化模式及码本hash，拒绝不兼容checkpoint。

#### Scenario: 错误来源
- **WHEN** 文件hash、形状、有限值或抽样SID不匹配
- **THEN** 在训练开始前明确报错

#### Scenario: 历史A
- **WHEN** 使用默认full_content加载既有A
- **THEN** 保持原checkpoint契约与行为兼容

### Requirement: 有边界的实验入口

根脚本 SHALL 经既有训练脚本及统一src.main入口，默认串行运行两个条件，使用Beauty seed42 GPU0/1，支持单条件、dry-run、两种notes语法及最后优先的额外Hydra override。系统 SHALL 支持显式指定两个不同物理GPU及独立master port，使两个单条件任务可并行运行。文档 MUST 区分初步筛选与机制确认。

#### Scenario: 两组双卡并行
- **WHEN** 用户分别以first_only/gpus=0,1/port=29750和deep_residual/gpus=2,3/port=29751启动
- **THEN** 两任务各启动两个进程，各自可见所选物理GPU，Trainer均使用映射后的逻辑devices=[0,1]，配置记录物理卡号，保持每任务训练预算及batch不变

#### Scenario: 双条件启动
- **WHEN** 用户执行默认命令
- **THEN** 执行first_only及deep_residual各20k steps，并记录结构化条件和参考run，任一失败立即停止

#### Scenario: 手动实验边界
- **WHEN** agent完成实现与轻量验证
- **THEN** 仅交付同步状态及启动命令，不自动运行完整实验
