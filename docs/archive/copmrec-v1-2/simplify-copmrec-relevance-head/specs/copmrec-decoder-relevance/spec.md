## ADDED Requirements

### Requirement: Decoder-only relevance residual
系统 SHALL 提供CoPMRec v1.2，只以读取完整候选SID后的decoder最后一层末位置hidden state作为相关性head输入；最终分数仍为content分数加该残差。

#### Scenario: Candidate relevance scoring
- **WHEN** 为用户候选评分
- **THEN** decoder输入为start与完整SID，head输入维度为d_model，h/v及h*v不直接进入head，用户条件通过cross-attention读取

#### Scenario: Joint training
- **WHEN** 从零训练v1.2
- **THEN** 推荐参数全部可训练，继承基础三项loss与权重0.05726763550972437的候选CE，不冻结teacher或推荐组件

### Requirement: Versioned execution and outputs
系统 SHALL 提供独立训练/推理入口，继承v1.1批量评分及候选、selection/audit协议，并严格校验版本、head输入、chunk及loss契约。

#### Scenario: Start scratch DDP training
- **WHEN** 使用v1.2根训练脚本以两进程启动且checkpoint引用均null
- **THEN** 通过uv run torchrun与src.main装配v1.2，记录独立版本/group，支持dry-run、notes及额外override

#### Scenario: Restore incompatible checkpoint
- **WHEN** 将v1.1 checkpoint作为v1.2恢复
- **THEN** 系统拒绝跨版本恢复，不忽略head形状或契约差异

#### Scenario: Trace validation
- **WHEN** 读取v1.2候选trace
- **THEN** 校验对应version、批量执行、chunk与decoder_after_complete_sid head输入，标签不参与推荐评分

### Requirement: Preserve prior versions
系统 SHALL 保持v0/v1/v1.1的既有参数结构、评分、初始化、入口及恢复契约。

#### Scenario: Prior checkpoint reuse
- **WHEN** 以匹配旧版本配置加载checkpoint
- **THEN** 使用原head维度与原分数计算，不切换为v1.2
