## ADDED Requirements

### Requirement: 固定辅助视图与联合目录评分

系统 SHALL 新增v5.3 `UnifiedNativeViewCECoPMRec`，由v5.2派生，无新constructor kwargs或参数。训练 SHALL 复用一次目录projection及其dropout结果，用同一个query计算无目录残差全目录CE，固定权重1；原SID CE、联合目录content CE和mixture各权重1、learned alpha及部署联合目录评分保持。辅助视图 SHALL NOT 被平均到部署logits，eval SHALL 不新增辅助loss键；新增`native_view_ce_loss` SHALL 仅在train配置MeanMetric／key日志映射。

#### Scenario: 随机起点与梯度解释

- **WHEN** seen residual初始为零
- **THEN** 两目录logits相同，content监督初始加倍；实际梯度可以经共享query／projection／history残差路径传播，不声称已隔离视图机制或保护全部native命中。

### Requirement: 版本和辅助监督契约

系统 SHALL 记录exact8 `native_view_ce`：protocol=`copmrec-shared-query-native-catalog-ce-v1`、weight=1.0、training_only=true、training_support=`all_catalog`、training_targets_must_be_seen=true、catalog_residual=false、shared_query=true、shared_projection=true。它 SHALL 与模型property、顶层配置、共享writer metadata及checkpoint内21keys `copmrec_unified_scratch`一致；继承v5.2 dense_ce_support exact5，版本为v5.3。

#### Scenario: 严格正常恢复

- **WHEN** 读取wrong version、aux weight或支持集契约checkpoint
- **THEN** 正常恢复拒绝；合法ownbest保持原URI／SHA／Artifact身份及完整连续optimizer／scheduler来源，不以last替换best。

### Requirement: 不变训练配方与单模型部署

系统 SHALL 用seed42随机初始化、全部模块共同从step0连续50000更新，无推荐checkpoint warm-start／teacher／pool。DDP2每卡128／global256／FP32、AdamW peak.0003及.002／WD.035、warm2500／horizon50000／min0／clip1、4层SID及原输入Artifact保持。own raw dense每500步N10首个最大值选best；独立部署 SHALL 使用单物理GPU→local[0]／单进程、原history资格及stable catalog row ties。

#### Scenario: 薄配置和统一脚本入口

- **WHEN** 通过新根train／inference脚本调用
- **THEN** 走既有`src.main`／Hydra统一入口，notes两形式、quoting、dry-run和后置override透传；训练拒绝外部checkpoint，推理必须显式own selected URI／SHA并拒多进程，默认不dry-run。

### Requirement: 显式追加有界成本

用户此次授权 SHALL 仅新增1train／50000更新＋1完整singleGPU Validation，新增Testing0／scan0；旧3train／150k及3完整Val与失败startup SHALL 保留，不重置。root SHALL 单独登记并根据实际job／source／CP／output计数，准备或测试不得算正式完成。

#### Scenario: 主门禁与停止边界

- **WHEN** 完整175原始文件／22363用户、labels／keys／输入／CP／source及合法唯一零history输出审计通过
- **THEN** 用固定同policy native42检验R10与N10各至少+8%、两个paired绝对CI95下界正，并单独描述v5.2增量；只有本单seed结果，不能宣称43配对／Testing或整体可复现goal完成。

#### Scenario: 负向或不确定结果

- **WHEN** 主门禁未过或相对v5.2无明确价值
- **THEN** 依实际CI／增量保留部分证据或停止该固定辅助CE，不自动追加lambda／CP／温度／seed／续训扫描；旧v5.2正向和v5.1负结果仍保留其原范围。
