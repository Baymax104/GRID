# 设计

以实时读取的7y54j4m6完整W&B配置作为原始v0训练参数依据，以原liger_joint_inference链路和testing数据协议恢复v0预测。v0仅添加版本/group/task名称元信息，不改变原模型目标、内容与生成机制、训练预处理、dense验证、hybrid预测、batch、AdamW/scheduler、50k更新或500验证频率。

v1与v1.1迭代已结束，将既有selection/audit、2500间隔和指标同步设置写到其独立组件/入口中，验证所有resolved行为配置不变，允许元信息引用的装配方式改变。

用户明确v2也统一evaluation/testing，因此v2使用相同FileDataModule及原始预处理/批量设置：全evaluation验证、testing推理，每500步验证；融合head仍只使用decoder最终状态，训练固定0.01候选CE，相关性修正固定0.5。v2以hybrid融合最终评分选best，这是其方法组件与v0 dense选点的明确差异。保留v2 DDP关闭buffer广播/分布式sampler与全局总和指标同步，避免恢复已知分片问题；配置核验不得悄然切换v2为dense content-only而绕过组件。

验证不启动完整实验或创建W&B run。原始配置快照与更改前八个版本resolved配置留存；按参数、数据、指标、回调与trainer比对，再做实际Bash参数替身和node1指纹核验。
