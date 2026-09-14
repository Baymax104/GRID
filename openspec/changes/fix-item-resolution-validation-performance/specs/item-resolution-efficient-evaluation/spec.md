## ADDED Requirements

### Requirement: 轻量验证保持推荐语义
系统 SHALL 在 validation 跳过不消费的搜索 trace 和概率证书，并保持完整搜索的推荐 SID、分数及预算策略。

#### Scenario: 验证与完整 trace 对比
- **WHEN** 同一模型和输入分别使用轻量路径与完整 trace 路径
- **THEN** 推荐 SID 和分数相同，轻量路径不计算最终上界或事件 trace

### Requirement: 批量概率证书
系统 SHALL 在完整 trace 中用批量目录索引计算每 item 上界，不逐前沿节点执行设备成员查询，且保持概率质量及 Top-K 证书含义。

#### Scenario: 多层前沿与空前沿
- **WHEN** 前沿含不同层的互不包含节点，或完全展开后为空
- **THEN** 批量上界等于逐节点参考实现，剩余质量与已解析质量之和为一

### Requirement: 实验与 checkpoint 兼容
系统 SHALL 保留旧 checkpoint contract、数据 split、验证周期、搜索预算和命令入口。

#### Scenario: 重新启动已有矩阵
- **WHEN** 使用既有两条队列命令
- **THEN** 条件矩阵仍为 54 次训练且验证每 500 步执行；派生索引不新增持久化 checkpoint 字段

### Requirement: 可核查的性能验收
系统 SHALL 通过输出等价和目录规模复杂度检查报告性能改善，不将 CPU 探针结果宣称为真实 GPU 加速。

#### Scenario: 宽目录搜索
- **WHEN** 有大量未展开合法前缀
- **THEN** 证书计算不产生与前沿节点数等量的单节点 members 调用
