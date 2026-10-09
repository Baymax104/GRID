# letter-pipeline Specification

## Purpose
将 LETTER 各阶段接入统一 Lightning/Hydra 入口和可追溯产物链路，明确上游路径、根脚本参数透传及 checkpoint 恢复，通过有限 GPU 验证确认训练和导出链路可运行。
## Requirements
### Requirement: 统一入口与显式上游
系统 SHALL 通过src.main和Hydra experiment运行四个阶段，使用显式embedding_path、cf_embedding_path、semantic_id_path和ckpt_path。
#### Scenario: 脚本参数透传
- **WHEN** 用户传入dry-run、非空notes和额外Hydra override
- **THEN** 脚本正确quote并将额外override放在默认参数之后。

### Requirement: 有界GPU验证
系统 SHALL 验证GPU上的dry-run、5step及checkpoint恢复，不自动开始正式实验。
#### Scenario: 本地CUDA不可用
- **WHEN** 旧隔离环境不存在
- **THEN** 系统使用node1既有CUDA环境，不重新安装PyTorch/CUDA。
