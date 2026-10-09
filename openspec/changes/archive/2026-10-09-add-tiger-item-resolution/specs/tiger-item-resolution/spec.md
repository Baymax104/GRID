## ADDED Requirements

### Requirement: 独立且归一化的物品解析

系统 SHALL 在独立推荐模块中实现合法目录的停止/路由概率，并对同 item 的多个互斥解析深度求和。

#### Scenario: 精确枚举目录
- **WHEN** 小目录完全展开
- **THEN** item 概率之和为 1，推理分数等于教师强制计算的边缘似然，且不存在非法或重复 item

### Requirement: 因果训练与匹配控制

系统 SHALL 提供九个指定 arm，前缀状态不得读取未生成后缀；匹配结构对照共享内容信息、初始化和局部容量。

#### Scenario: 改变目标后缀
- **WHEN** 历史与前缀相同而目标后缀不同
- **THEN** 相同前缀的状态、解析分布与 gate 相同

### Requirement: 可审计预算与身份

系统 SHALL 保存目录/方法指纹，拒绝不兼容 checkpoint；预算推理 SHALL 保存全部未展开质量并给出有效的 item 下界。

#### Scenario: 预算耗尽
- **WHEN** 推理只展开部分前缀
- **THEN** 已输出总质量与剩余质量之和为 1；不足 K 个唯一正质量 item 时明确失败

### Requirement: 新 trace 共享输出

系统 SHALL 通过共享 writer 发布独立 schema 的 item resolution trace，保留 user keys、labels、预算与方法元数据。

#### Scenario: 合并输出分片
- **WHEN** writer 合并多个预测 batch
- **THEN** keys 唯一、trace 行对齐且经过新 schema 校验，旧 prefix writer 默认行为保持兼容

### Requirement: 训练集标定与适配声明

系统 SHALL 将 WIDE 风格阈值限制为 training split 标定，并核对 checkpoint/catalog 身份；COBRA/WIDE SHALL 标记为适配控制。

#### Scenario: 错误标定来源
- **WHEN** 使用 evaluation/testing 或不匹配 checkpoint 的标定产物
- **THEN** 拒绝 WIDE 推理而不是静默使用阈值
