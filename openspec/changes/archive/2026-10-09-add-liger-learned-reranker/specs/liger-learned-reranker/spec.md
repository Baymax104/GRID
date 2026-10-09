## ADDED Requirements
### Requirement: 标签无关的匹配候选缓存
系统 SHALL 使用冻结模型创建joint/control两臂，候选数量逐用户相等，包含同一cold集合和dense Top10，特征不读取目标标签。
#### Scenario: 更换标签
- **WHEN** 同一历史使用不同标签
- **THEN** 候选、评分、特征不变，仅监督位置变化
### Requirement: 分片来源与完整性
系统 SHALL 保存标准keys/predictions分片、来源模型SHA、catalog指纹与分片SHA，拒绝不完整缓存、重复用户或错误split。
#### Scenario: 文件被修改
- **WHEN** 分片哈希不符
- **THEN** 缓存读取失败
### Requirement: 共同训练和隔离
系统 SHALL 只用training缓存、共同覆盖且非留出用户拟合两臂；归一化不得使用内部验证或evaluation数据。
#### Scenario: 目标池外
- **WHEN** 任一臂未覆盖训练目标
- **THEN** 两臂训练均排除此样本，评价保留该用户
### Requirement: 受限残差训练
系统 SHALL 实现263→64→1零输出初始化MLP，s=d+tanh(r)，只训练该MLP并持久保存归一化、arm及来源身份。
#### Scenario: 零初始化与checkpoint错用
- **WHEN** 新初始化或加载另一arm的checkpoint
- **THEN** 前者复现dense Top10，后者报错
### Requirement: 统一运行与手动预算
系统 SHALL 经src.main和Hydra执行缓存、训练、评价，支持dry-run和notes及额外override，dry-run不发布结果。
#### Scenario: dry-run
- **WHEN** 用户显式传入--dry-run
- **THEN** 统一入口仅执行最小批次并禁用产物写入

### Requirement: 缓存默认本地存储
系统 SHALL 默认仅将可再生成特征缓存写入运行机器本地，不上传W&B；保留运行指标、manifest和哈希，下游优先读取本地manifest。
#### Scenario: 默认缓存生成
- **WHEN** 未明确要求发布缓存
- **THEN** writer及配置关闭publish_wandb，W&B运行指标仍记录
