## ADDED Requirements

### Requirement: 五个受控训练变体

系统 SHALL 提供 A1 无 mixture、A2 无历史/目录残差、A3 无 native、A4 合法生成替换 mixture、A5 joint CE 替换 native 的独立 scratch 实验，保留 50k/DDP2/global256/FP32 和原选点/部署。

#### Scenario: 恢复变体 checkpoint
- **WHEN** 使用其他变体或正式 Full checkpoint 恢复消融
- **THEN** 系统拒绝；正确变体完整状态可恢复，A1/A4 gate 和 A2 残差按契约冻结

### Requirement: 三个机制证据包

系统 SHALL 通过统一 Trainer.test 提供 keyed 命中/排名分解、固定 checkpoint 残差位置干预、真实前缀条件概率测量，且不更新权重或改变部署。

#### Scenario: 固定残差位置干预
- **WHEN** 关闭历史残差
- **THEN** 重新编码 query，目录干预与历史资格单独控制，输出四视图及可复算用户级观测

#### Scenario: 真实前缀测量
- **WHEN** 使用 Full/A1/A4 的 own-best
- **THEN** 输出合法条件概率/NLL/熵/JS/目标rank，A1/A4 的 mixed 字段为空，标签不进入正式预测

### Requirement: 脚本与命令契约

系统 SHALL 提供根目录启动脚本，支持 dry-run、两种 notes 写法与额外 override；训练两个进程，推理/诊断一个进程。所有新 issue SHALL 使用实际入口与统一 W&B group 格式。

#### Scenario: 不完整命令输入
- **WHEN** 变体、checkpoint、哈希、路径或资源输入为空或错误
- **THEN** 脚本在启动前拒绝，尚未产生的 checkpoint 使用显式待填占位且不得替换成 Full

### Requirement: Full 保持数值与契约

系统 SHALL 保持正式 Full 的参数、随机数、损失、两组优化器及恢复契约。

#### Scenario: 相同初值与内存 batch
- **WHEN** 执行正式模型原验证
- **THEN** 预测、四损失及保存恢复与本变更前一致
