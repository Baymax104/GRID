## 背景与五项判断

1. 核心假设：共享协同残差的单模型中，给同一query面对无目录残差视图的目标竞争直接CE，可能保留部分语义覆盖并改善seen推荐取舍。该假设可被整体／增量结果反驳，不把原因预先定为残差“捷径”。
2. 支持：v5.2相对native的部分正向与seen命中互补已经实际审计；cold已不支持继续作为主要覆盖瓶颈。新增辅助视图的效果尚未运行。
3. 反证与缺口：v5.1固定评分平均对v5双指标CI为负，仅否定该平均评分，不是本辅助CE的隔离对照；v5.2未过双8。新方案没有第二seed／Testing效果证据。
4. 最小鉴别：唯一固定v5.3与已保存v5.2／native42同policy全量输出比较；无阻断实现疑点后不追加lambda／query／CP矩阵。零残差初始两CE相同，单臂不能分离视图监督与content CE权重加倍的贡献。
5. 成本：用户明确新增1随机起点连续50k训练＋1singleGPU完整Val、0Test／0扫描。旧3train／150k与3Val保留封口；新累计4train／200k、4完整模型Val，不重置旧失败startup或授权43配对。

## 固定实现

新增`src.recommendation.liger.unified_native_view_ce.UnifiedNativeViewCECoPMRec`，继承v5.2，无新增constructor kwargs／参数，版本v5.3。原decoder和单次目录projection的调用／dropout结果复用；联合目录logits继续用于原content CE、mixture与部署，辅助目录为`normalize(projected_content)`，复用同query／projection，仅增加一次matmul和全目录CE。

训练目标：`SID_CE + joint_content_CE + mixture_NLL + native_view_CE`，辅助权重固定1，原三loss各1。history仍含共享seen残差，所以辅助CE会经query和history更新残差；“无残差”只描述辅助目录，不是独立native LIGER、teacher或参数保护。

`native_view_ce` exact8：

```yaml
protocol: copmrec-shared-query-native-catalog-ce-v1
weight: 1.0
training_only: true
training_support: all_catalog
training_targets_must_be_seen: true
catalog_residual: false
shared_query: true
shared_projection: true
```

保留v5.2 `dense_ce_support` exact5；`unified_scratch_contract`为原19＋dense支持集＋native视图共21keys，checkpoint字段仍`copmrec_unified_scratch`，错版本／错权重／错契约拒绝。训练返回新增`native_view_ce_loss`，eval无额外loss键；正常部署同一固定参数的评分与v5.2相同。

model配置薄继承v5.2，仅换target并增加train-only `native_view_ce_loss` MeanMetric／key映射，其余参数不变；val／test没有该metric，避免依赖不存在的eval返回字段。train／inference experiment继承v5.2，声明独立task/group/notes、v5.3版本及native字典；训练checkpoint writer与推理artifact writer保留共享协议。推理真实版本字段为`checkpoint_identity.version`，没有额外outer `copmrec_version`。实际21keys由CPU模型属性回填，不能手写伪造。

## 训练与部署条件

seed42／随机起点无推荐checkpoint，DDP2每卡128／global256／FP32，AdamW主干peak.0003／residual.002／WD.035，warm2500／horizon50000／min0／clip1，单次连续50000、4层SID、原causal32／有效history20／输入Artifact不变。每500步own raw dense Val N10首个最大值选best；budget与实际保存fullstate步数分别记录，last不替代best。

独立完整Val单物理GPU→local[0]／单进程，单checkpoint正常恢复；原history资格、stable catalog row ties、cold eligible，175文件／22363用户／真实labels／keys／输入／CP／source和合法唯一零history输出先审。固定native42与v5.2输出复用；主要门禁仍相对native R10／N10各≥8%且paired绝对CI95下界正（PCG64 seed42、2000次）。不按Testing选模型。

## 取舍与风险

若主门禁通过，冻结单seed模型，披露43配对／Testing尚缺，后续成本另明确授权；不宣布整个可复现目标完成。未过但有清楚部分收益则保留对应证据，CI跨0或微小增量只描述不确定；相对v5.2无明确价值则停止本固定辅助CE，保留v5.2，不自动扫权重／续训／改名维持假设。

新增训练matmul／CE／反向但不增加部署计算；相同步数／参数不等于同FLOPs、GPU时长或搜索成本。单seed开发集、逐点CI与native42历史训练source缺口保持，不由新归档回填。旧402运行字节须逐项保留。

## 验证

核心只覆盖实际gradient／单次projection和RNG顺序／零残差相等／eval及部署等价／完整契约恢复与错版本拒绝。配置完成完整Hydra resolve、parent训练条件逐项相等、共享writer纯dict metadata及Bash语法／notes两形式／quoting／空错误值／override／dry-run／单卡guard。正式任务由root在source／同步／CPU零optimizer和DDP2一步smoke通过后唯一启动，完整产物终态后审计；本文档创建不启动实验。
