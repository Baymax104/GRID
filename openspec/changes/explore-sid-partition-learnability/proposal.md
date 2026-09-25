## Why

已有固定SID路线未建立可修复机制；量化候选需要先确认是否存在超出频次的、可复现的行为条件预测信号，避免直接启动六个推荐训练run。依据研究库的全实验路线审查与SID量化问题筛查，首阶段只验证代理资格，不声称量化瓶颈已成立。

## What Changes

- 新增仅读取training的CPU统计探针，经统一入口和Trainer.test运行。
- 固定首层分组、末次训练转移、四个互斥用户集合、平滑条件计数及频次/置乱对照。
- 输出可审计的用户级、分组级统计、输入哈希、置信区间、可靠性与停止判定。
- 提供根脚本、dry-run、notes和Hydra override；正式运行由用户手动启动。
- 写明后续分组干预和冻结复现协议，但不实现或启动后续模型训练。

## Capabilities

### New Capabilities

- `sid-partition-learnability`: training内分组行为信号的资格诊断。

### Modified Capabilities

无。

## Impact

新增独立data helper/datamodule、quantization分析组件、component配置、根脚本和测试；复用Artifact loader、lineage和StructuredAnalysisWriter，不修改既有推荐模型，不新增依赖。同步research协议与状态，保留G1/CCFD等暂停决定。
