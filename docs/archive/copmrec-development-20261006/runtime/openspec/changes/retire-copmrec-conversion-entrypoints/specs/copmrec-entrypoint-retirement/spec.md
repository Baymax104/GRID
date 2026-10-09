## ADDED Requirements

### Requirement: 旧收益转化入口退役

系统 SHALL 将 manifest 中的 10 个根脚本与 9 个 experiment 配置移出标准可启动范围，保留基础 CoPMRec 与 baseline 入口。

#### Scenario: 选择旧 experiment
- **WHEN** 从仓库根目录 compose 已退役的 experiment 名称
- **THEN** Hydra 报告配置缺失，不能启动旧阶段

#### Scenario: 选择基础版本
- **WHEN** compose liger_joint_train/inference 或 liger_train/inference
- **THEN** 配置仍指向原模型及原协议，清理前后的核心文件哈希一致

### Requirement: 退役文件可追溯

系统 SHALL 在删除前保存入口与旧启动测试的原始字节，并记录逐文件路径、大小和 SHA256，以及 ZIP SHA256。

#### Scenario: 检查归档
- **WHEN** 读取退役 manifest 和 ZIP
- **THEN** ZIP 内全部文件与清理前内容的大小/SHA 相同，源路径已退出标准入口或测试收集范围

### Requirement: 保留算法及运行资产

系统 SHALL 保留旧算法实现、内存算法测试和所有实验资产，使用项目 Mutagen 入口传播受管范围删除。

#### Scenario: 同步验证
- **WHEN** 清理检查完成并执行 mutagen_sync.ps1 flush/status
- **THEN** 三个会话均 Watching 且无 conflict；远端旧入口不存在、基础入口仍存在，不通过远端 Git 判断同步状态
