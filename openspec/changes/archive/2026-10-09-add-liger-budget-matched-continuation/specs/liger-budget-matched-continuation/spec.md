## ADDED Requirements

### Requirement: Native LIGER仅权重三段训练

系统 SHALL 通过统一src.main运行phase A/B/C上限6000/6000/2000；原SID/content两个loss保留，source为原native LIGER Valbest45000、nativeA best、nativeB best的顺序链。

#### Scenario: 读取阶段前驱
- **WHEN** 开始一个正式阶段
- **THEN** 公共loader解析固定URI，核SHA256、nativehook、catalog和完整strictstate，记录实际前驱步数，不能恢复optimizer/scheduler/global step。

#### Scenario: 拒绝错误来源
- **WHEN** 来源为CoPMRec、错误phase/URI/SHA/catalog、非有限或不完整state
- **THEN** 启动失败，不以warning或缺省last.ckpt继续。

### Requirement: 明确匹配训练与选择日程

正式运行 MUST 使用freshAdamW LR1e-4、WD.035、warm300、cosine horizon6000/min0，global256、seed42、FP32、clip1、每1000步验证；A/B原dense无history排除选ValNbest，C为historyexcluded dense ValNbest。

#### Scenario: 短末段仍保留原horizon
- **WHEN** phase C执行2000 updates
- **THEN** scheduler horizon保持6000；actualterminal2000与selectedbest分别验证并记录。

### Requirement: 真实native双成员公平pool

系统 SHALL 从真实A与C checkpoint恢复两个成员，验证共享祖先身份，各独立编码后固定0.5 fullcataloglogits平均，再按同有效输入history资格和稳定row排序Top10；仅允许单进程推理。

#### Scenario: Pool与单成员对照
- **WHEN** 执行最终Validation
- **THEN** singleC与poolAC各一次，均使用固定data/raw/catalog/历史窗口，不读取label调score，不扫描pair/weight。

### Requirement: 可审计统一入口与源字节

配置脚本 MUST 使用根目录统一Hydra入口、支持显式dry-run/notes/额外override，双卡训练与单卡推理；正式运行归档实际runtime字节及dirty/untracked文件，来源和checkpoint消费通过共享loader/callback记录。

#### Scenario: 运行前核验
- **WHEN** 正式阶段launch
- **THEN** officialMutagenflush三会话Watching无conflict、预登记所有runtimehash与实际远端一致、sourceorigin验证且CPU恢复及smoke通过；唯一job句柄先保存。

### Requirement: 新增有界成本及结果判定

新增阶段 SHALL 限定3训练/14000updates、2完整Val、最多1新Test，原caps不重置。唯一baseline以ValN→R→简单方案选择，CoPMRec已有固定Test输出复用；完整生产成本64k及selected祖先长度分开披露。

#### Scenario: 匹配后收益不足
- **WHEN** 更强baseline使原双10%不成立或pairedCI跨0
- **THEN** 报告真实差值和边界并收缩主张，不用Testing换CP/方法或自动增加预算。
