## 1. 实现
## 7. NDCG分母training均值上限
- [x] 7.1 实现可选上限、training拟合、冻结目标契约与audit兼容
- [x] 7.2 梯度/数据边界/恢复契约/配置透传聚焦测试及OpenSpec验证
- [x] 7.3 node1同步、真实缓存与实现探针核验、固定阶段及手动训练命令

## 5. 内容正确关系保护
- [x] 5.1 单侧rank-discount保护损失、日志与默认关闭配置
- [x] 5.2 checkpoint严格契约与旧臂/audit兼容
- [x] 5.3 聚焦梯度/恢复监督/配置测试和OpenSpec验证（26项通过）
- [x] 5.4 node1同步、training小批量鉴别、阶段记录和手动命令（32用户，0更新/0新run）

- [x] 1.1 历史与候选join、来源契约
- [x] 1.2 可微混合评分、decoder冻结与两臂目标
- [x] 1.3 统一双checkpoint audit及兼容trace
- [x] 1.4 三入口配置和脚本
## 2. 验证交付
- [x] 2.1 聚焦测试及配置/脚本检查（57项通过）
- [x] 2.2 OpenSpec strict
- [x] 2.3 node1同步与真实历史轻量核验（flush成功，16运行文件哈希一致，training两臂梯度/selection评分通过）
- [x] 2.4 冻结协议、预算及手动命令

## 3. 2026-10-02效率修复
- [x] 3.1 NLL仅解码已覆盖目标，冻结目录投影缓存，增大候选chunk，CPU线程4
- [x] 3.2 损失/梯度等价、配置及node1有限小批量吞吐核验（24项测试、约7倍局部加速）
- [x] 3.3 同步、保留原运行记录并交付更新命令与协议说明（checkpoint与optimizer恢复契约通过）

## 4. 2026-10-02有效竞争pair
## 6. 双分支正确关系保护
- [x] 6.1 teacher参照最大并集、默认兼容与目标契约
- [x] 6.2 聚焦梯度/重复pair/配置及OpenSpec检查
- [x] 6.3 node1同步、training梯度鉴别与固定阶段命令

- [x] 4.1 实现content/current mixed Top20并集的负例mask，分子分母一致，保留all旧目标
- [x] 4.2 配置及checkpoint目标契约，兼容旧checkpoint与NLL audit
- [x] 4.3 聚焦损失/梯度/配置测试与OpenSpec strict（21项通过）
- [x] 4.4 node1同步、有限training batch梯度检查、冻结新阶段与手动命令（8用户，0更新/0新run）

## 8. Top10边界辅助项
- [x] 8.1 32 training用户零更新参数梯度鉴别
- [x] 8.2 默认关闭边界loss、training-only校准与checkpoint/audit契约
- [x] 8.3 聚焦梯度、校准、配置/脚本及兼容测试和OpenSpec strict
- [x] 8.4 node1同步、生产探针、来源与手动训练命令交付
