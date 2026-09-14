## ADDED Requirements

### Requirement: Independent keyed catalog bank
系统 SHALL 在独立模块中按 item key 对齐完整 SID 与有限内容向量，构建固定投影和含计数的稀疏前缀原型，不修改 TIGER baseline。

#### Scenario: Permuted input bundles
- **WHEN** SID 与 embedding bundle 的 item 顺序不同
- **THEN** 内容按 key 正确重排；缺失 key 或重复完整 SID 被拒绝

#### Scenario: Prototype semantics
- **WHEN** 一个前缀包含多个 item
- **THEN** 原型计数之和等于子树 item 数，均值为实际 cluster 均值，半径为最大内容距离

### Requirement: Consistent train and decode scoring
系统 SHALL 使用共享 query 的计数加权分支质量、合法分支归一化和有界混合，训练与生成使用相同条件概率。

#### Scenario: Leaf branch
- **WHEN** 每个子分支仅包含一个 item
- **THEN** 分支内容分数等于该 item 的点积分数，非法分支概率为零

#### Scenario: Backward and generation
- **WHEN** CGBS 执行一次训练反传及 beam generation
- **THEN** 所有启用参数梯度有限，输出唯一有效 SID，概率与 trace 一致

### Requirement: Explicit experimental controls
系统 SHALL 提供 original、mask_ce、token_content_init、single_prototype、full、no_aux、shuffled、hybrid，并在配置记录条件。

#### Scenario: Original and shuffled
- **WHEN** 用户选择 original 或 shuffled
- **THEN** original 复用 baseline 行为；shuffled 仅固定打乱目录内容对应关系而不改变 backbone 随机初始化

#### Scenario: Hybrid output
- **WHEN** 用户选择 hybrid
- **THEN** 最终推荐来自生成与稠密候选的合并排序，且不能输出冒充最终候选过程的 beam trace

#### Scenario: Hybrid validation diagnosis
- **WHEN** 用户对无 trace 的 Hybrid evaluation 推荐运行 diagnosis
- **THEN** 独立适配层显式读取 evaluation 标签，不能默认切换到 testing；提供的 trace 必须与该划分一致

### Requirement: Checkpoint provenance
系统 SHALL 保存固定 bank 及其身份、条件和算法参数；恢复时 MUST 拒绝不匹配的来源或条件。

#### Scenario: Changed catalog
- **WHEN** 恢复 checkpoint 时目录内容或 SID 发生变化
- **THEN** 在训练或推理之前显式报错

### Requirement: Reproducible manual experiment launch
系统 SHALL 通过统一 Hydra 入口提供训练、推理及两组 GPU 队列；data-dir 必填，seed 默认为 42，notes 双形式可用，额外 override 最高优先级，支持 dry-run。

#### Scenario: User starts a queue
- **WHEN** 用户手动启动第一组或第二组实验
- **THEN** 指定 GPU 上顺序运行相应六个实验，共享数据及选优协议，不自动启动另一组

#### Scenario: Missing required data or conflicting override
- **WHEN** 必填路径为空或用户提供末尾 Hydra override
- **THEN** 空路径在启动前被拒绝；有效末尾 override 覆盖脚本默认值
