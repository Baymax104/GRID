## ADDED Requirements

### Requirement: 可复核的冻结评分诊断

系统 SHALL 在显式开启 audit training diagnostics 时记录冻结候选来源、当前 base、最终全目录分数、候选 item keys、真实目标及派生排名、CE、候选外竞争和残差指标；预测排名 SHALL 不受标签影响。诊断 SHALL 复用 reference 分数且不计入原有搜索策略耗时。

#### Scenario: dense 候选内改善但目录外退化
- **WHEN** 固定候选中的目标分数提高、候选外 item 分数提高更多
- **THEN** 证据能同时显示候选 CE 下降、全目录 CE/目标排名恶化以及候选外竞争增加

#### Scenario: 默认兼容与损坏证据
- **WHEN** 未开启诊断或诊断分数、候选、排名证据被破坏
- **THEN** 默认继续产出 v1，v2 校验器拒绝非有限值、非法候选、错误排名和与 reference Top-K 不一致的分数

### Requirement: 初始化与候选来源一致

系统 SHALL 验证三个分支在零更新、eval 模式下与同一 base checkpoint 分数一致，冻结候选来源不随 dense 更新变化。GPU 诊断中的零残差分数 MUST 标记为反事实，不能冒称重新运行的初始化实验。

#### Scenario: 未更新的新分支
- **WHEN** base checkpoint 初始化 dense、prefix_free、brir，尚无 optimizer step
- **THEN** 全目录分数、稳定 Top-K 与共享候选一致；关闭诊断不改变预测结果

### Requirement: 有界手动实验入口

系统 SHALL 提供默认 4×128 evaluation 用户、单 GPU、固定已完成来源的 BRIR 手动套件，并交付 Sports 内容初始化的 20k steps、有效 batch256、GPU 0/1 命令。套件 MUST 支持 notes 两种形式、dry-run、print-only、错误参数拒绝及额外 Hydra override 透传。

#### Scenario: 预览与单 arm 重跑
- **WHEN** 使用 print-only 或选择单个 arm
- **THEN** 分别仅打印命令不执行实验，或仅执行指定 arm；所有实验仍从 src.main 装配并记录 W&B 来源
