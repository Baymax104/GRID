## ADDED Requirements

### Requirement: Exact prefix membership
查询 SHALL 返回与逐 token catalog 比较相同的布尔结果；安全整数范围内 SHALL 使用精确前缀索引，不改变候选顺序、分数或 beam 选择。

#### Scenario: Legal integer queries
- **WHEN** 查询合法范围内的不同深度 SID，catalog 可重复或乱序
- **THEN** 每项结果与原始比较一致

#### Scenario: Unsafe encoding
- **WHEN** radix 编码超出 int64、catalog 含非法 token 或候选为浮点类型
- **THEN** 使用兼容比较路径，保持原始判定

#### Scenario: Invalid candidate tokens
- **WHEN** 候选 token 超出 [0,K) 且其编码可能与合法前缀碰撞
- **THEN** 候选不能误命中合法 catalog

#### Scenario: Empty inputs
- **WHEN** catalog 或候选为空
- **THEN** 返回候选长度的布尔结果，空 catalog 的结果全部为 false

### Requirement: Cache and checkpoint compatibility
索引 SHALL 不进入 checkpoint，并 SHALL 在 catalog 正常替换、原地修改或设备迁移后更新。

#### Scenario: Catalog changes
- **WHEN** 已构建索引后 catalog 被替换或原地修改
- **THEN** 后续结果反映当前 catalog，包括不跟踪版本的 inference tensor

#### Scenario: Legacy checkpoint
- **WHEN** 构建索引后 strict 加载原 checkpoint
- **THEN** 无新增缺失或多余的 state_dict 键
