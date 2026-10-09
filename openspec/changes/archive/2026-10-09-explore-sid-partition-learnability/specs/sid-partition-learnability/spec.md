## ADDED Requirements

### Requirement: Training-only deterministic inputs
系统 SHALL 仅使用显式training目录、按key对齐的SID与embedding，按用户固定抽样并划分四个互斥集合。

#### Scenario: Duplicate compatible histories
- **WHEN** 同用户出现兼容前缀记录
- **THEN** 系统取最长记录且与输入顺序无关，只贡献一次末转移

#### Scenario: Invalid data identity
- **WHEN** 出现未知item、非整数SID、不兼容用户记录或非training目录
- **THEN** 系统报错且不生成成功证据

### Requirement: Held-out conditional signal
系统 SHALL 使用fit-only平滑条件计数、频次基线、99个保边际context置乱及用户bootstrap评价两次互斥重复。

#### Scenario: Unseen context
- **WHEN** eval上下文未在fit出现
- **THEN** 条件预测回退频次基线，用户gain为零

#### Scenario: No held-out leakage
- **WHEN** eval目标变化
- **THEN** fit模型及频次分布不变

### Requirement: Bounded group reliability verdict
系统 SHALL 输出控制支持度、目录组大小及语义离散度后的组可靠性，区分proxy_qualified、insufficient_support、no_go_for_this_proxy与smoke_only。

#### Scenario: Low support
- **WHEN** 合格组数或覆盖率不足
- **THEN** 返回insufficient_support而不宣称机制无效

### Requirement: Unified execution and evidence
系统 SHALL 通过src.main、Trainer.test、共享writer与lineage输出输入指纹、参数、逐用户和逐组统计，不新增离线入口。

#### Scenario: Dry run
- **WHEN** 用户显式传--dry-run
- **THEN** 限制读取和计算，关闭业务writer/logger，返回smoke_only

#### Scenario: Launcher overrides
- **WHEN** notes含空格引号或额外Hydra override与默认冲突
- **THEN** 参数保持字面值，额外override最后生效，缺失必填或非法参数被拒绝
