## ADDED Requirements

### Requirement: 仅目录侧使用residual
v5.4模型MUST从full-catalog CE scratch模型派生，在history编码中取消residual加法，在单联合目录评分中保留seen catalog residual，并保持原三项unit loss及learned alpha。

#### Scenario: 非零residual表
- **WHEN** 其他参数固定、seen residual表非零
- **THEN** history编码MUST不使用该表，catalog logits MUST保留它，cold residual为零

#### Scenario: 从头共同优化
- **WHEN** 推荐训练开始
- **THEN** 所有模块MUST从随机初始化、零gate与residual共同开始，沿固定连续50000更新日程训练，不使用其他推荐checkpoint或optimizer重启

### Requirement: 严格的residual placement checkpoint身份
模型MUST声明v5.4及exact3 placement契约：protocol为copmrec-catalog-only-residual-v1、history_residual为false、catalog_residual为true。residual与unified scratch两处sharing契约MUST均声明catalog_after_content_projection。

#### Scenario: 恢复不兼容的history语义
- **WHEN** checkpoint声明history/catalog共享残差、其他版本、placement缺失／多字段或bool字段使用非bool类型
- **THEN** 恢复MUST在将其权重接受为此模型前失败

### Requirement: 保持现有运行行为
实现MUST仅新增薄类，不修改原408 runtime文件或新增参数，并保持零residual时的catalog projection调用、初始化及RNG行为。

#### Scenario: 零residual比较
- **WHEN** v5.2与v5.4参数相同、residual为零且随机状态相同
- **THEN** query、catalog分数及joint losses MUST一致，旧模型行为MUST保持

### Requirement: 统一可执行配置和元数据
训练与推理MUST使用src.main的Hydra experiment及根脚本，透传notes、dry-run和后置overrides，并在记录配置与writer metadata一致声明v5.4 residual placement。

#### Scenario: 明确启动override
- **WHEN** 用户提供notes、dry-run及其他Hydra overrides
- **THEN** 脚本MUST保持quoted值、向统一launcher透传dry-run并允许后置override覆盖默认值

### Requirement: 不隐式新增实验额度
本地实现MUST NOT分配或启动新正式训练、完整Validation、Testing、seed43 pair或扫描。已完成4train／200000更新／4Val的账本MUST保持，直到收到明确新增额度。

#### Scenario: 额度答复待定
- **WHEN** 下一固定试验没有明确新增分配
- **THEN** 模型效果MUST保持未验证、正式运行MUST保持零、目标MUST NOT标为完成

### Requirement: 保持完整效果和复现要求
任何获授权固定试验MUST使用随机起点50000更新、own raw dense验证选择checkpoint及单GPU history-eligible完整目录Validation，保持同seed native双8及两paired绝对CI下界正的门禁。

#### Scenario: 取得seed42结果
- **WHEN** 实际候选Validation通过门禁
- **THEN** 仅此seed42模型MAY冻结，配对复现和独立Testing MUST明确保持未完成，直到获得真实证据

### Requirement: 专用授权与实际审计绑定
v5.4正式编排与审计MUST绑定其实际新增额度问题、真实答复及新注册，不接受已消费的v5.3授权。模型身份MUST按完整字段内容与类型判定，不以21键数量相同替代placement、sharing和version检查。v5.2描述性参考MUST保持其自身原合同。

#### Scenario: 没有新的明确答复
- **WHEN** 仅存在v5.3批准记录或v5.4请求记录，尚无v5.4真实批准及新注册
- **THEN** 正式同步、preflight、smoke、launch及audit MUST在远端调用或正式job写入前失败，纯本地fixture MUST NOT作为成功凭据

#### Scenario: 相同字段数的不同checkpoint语义
- **WHEN** 旧v5.3 CP保留native_view_ce且不含v5.4 placement，或v5.2参考仍采用原history+catalog sharing
- **THEN** 前者MUST被拒绝为v5.4，后者MUST按冻结v5.2契约核验，不能改写其sharing以通过
