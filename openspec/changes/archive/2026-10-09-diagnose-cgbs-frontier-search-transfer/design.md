## Context
来源A dragtsrn/39000，global ezvfov9u/500，branch 4yagd8r8/500。当前组合training_negative_no_advance保留。
## Goals / Non-Goals
一次人工启动作业，0训练，固定256用户及seed20260922；不扫alpha、宽度、checkpoint或testing，不宣称总体统计收益或论文新颖性。
## Decisions
- 复用按seed/user key最小SHA256抽样，evaluation全流只做CPU样本选择，模型仅处理256用户；batch1、单进程、FP32。dry-run处理2用户且不写正式产物。
- 来源A和两头通过共享artifact loader读取，要求global/branch契约、step500、源SHA、缓存SHA匹配，主干逐张量不变。一个模型切换冻结头；所有前向no_grad。
- 每用户运行原A beam、每头off与零残差beam、每头真实beam，以及共用A frontier构造/两头重评分。最坏8条beam遍历/用户（含A frontier构造），不跑全目录decoder。
- 原A与零残差有序SID必须精确相同，log分数误差固定atol1e-4/rtol0；不放宽容差。off也必须精确复现。失败仍记录原始flags/error，但summary判implementation_audit_failed，禁止因果分类。
- 固定候选逐层记录gold是否原已在A展开集合（由前层A存活判断）、补入标志、CE、严格/非严格rank、best-competitor margin、目标残差和候选间残差标准差。不能用带gold插入的局部统计证明真实可达。
- 实际搜索复用生产search，对比目标逐层存活及同层选中prefix集合相对A的重合；标签只用于诊断，不用于真实搜索打分或选择。记录共同命中排序与新增/丢失命中，不在256用户上宣称全量显著性。
- test_step只产出CPU记录；writer在test_end校验完整数量、唯一key、schema后原子写report.json、users.json及逐层CSV并可发布W&B Artifact。部分结果不发布正式report。
## Risks / Trade-offs
FP32近并列可能导致零残差排序不一致，此时保留实现有效性失败，不通过换排序规则修饰。256用户只能定位局部迹象；A/off算法一致不能证明所有评分校准正确。
## 决策与停止条件
有效性失败仅调查可复现bug；有效性通过且局部改善伴随实际路径受损则停止此目标/在线搜索组合；样本不足以区分原因则保持不晋级，不自动扩样本。最多本次1诊断作业、0新增训练。
