## Context

已有 A=`token_content_init`，Beauty seed42、20k steps 的参考 run 为 `5g3wpbg7`。本轮只新增两个训练条件，复用既有 A；参考结果由历史核查获得，训练后需重新核对 W&B 配置和 checkpoint 身份。

## Goals / Non-Goals

目标：判断深层完整内容初始化是否必要、是否优于匹配后的残差质心方案。非目标：不恢复 C/D/E，不改损失、预算、数据或搜索，不复现预训练 LLM 论文，不自动启动训练或推理。

## Decisions

### 五seed testing冻结协议 v1

根脚本tiger_sid_initialization_testing.sh复用标准catalog inference入口。默认all seeds、both条件顺序执行十个单卡任务；可显式单seed/单条件续跑，每个任务固定run和精确checkpoint文件。固定testing、beam10、arm=token_content_init，上游SID/content保持原身份，残差使用原训练推理组件，A使用历史兼容默认组件。发布keyed prediction和prefix trace，结构化记录协议、训练seed、训练run与预期checkpoint digest。预期digest是审计字段，不冒充底层下载强制校验；汇总必须核对实际lineage。任务名包含条件与seed，任一失败立即停止，支持notes/dry-run及最后优先Hydra override；override偏离冻结协议的结果必须排除正式比较并报告，不能静默纳入。

结果按每seed同用户key/标签比较，检查合法且唯一的10项预测、目录fingerprint与trace一致。主指标NDCG@10、辅助Recall@10，报告全部五个seed和等权seed平均差、胜负与leave-one-seed-out均值。用户区间以同目标前两层SID聚类，bootstrap 2000次seed42；跨seed固定样本t区间另列为探索性，不以用户数替代5个训练重复。全部模型完成后统一查看testing，不用它改checkpoint/beam/方法。历史W&B检查只能支持“未发现已使用记录”，不能证明未记录的离线使用不存在。

### 深层范数校准候选 v1

新增deep_norm_calibrated模式，仅允许A arm。第2、3语义层将均值mu变为mu*max(norm(mu),1e-8)^(eta-1)，再按校准表population std匹配原随机表std。eta为有限实数且0<=eta<=1；默认1，其分支跳过径向运算，逐位还原A。非校准模式不允许非1指数。实验组件固定eta=0.5。零均值保持零，低于1e-8的近零向量使用截断范数限制放大，截断区不声称精确幂律。保留首层、去重、未用码、参数数量和全局随机流。独立checkpoint契约记录版本、eta、范数下限；默认A契约不变。复用seed45 kckgwr6j与seed46 v7q7y0jx，预算20k；两seed均改善且后段不只峰值获益才考虑补齐五seed，混合结果不自动扫描eta。完整实验保持手动启动。

### 两个条件与默认兼容

保留 arm=`token_content_init`，增加 `token_initialization` 配置。`full_content` 默认逐字保持 A 的数学运算和随机流；`first_only` 仅初始化第1层；`deep_residual` 第1层保持A，第2、3层使用真实 RKMeans 质心。去重层及未占用码始终保持原随机值。非默认配置只允许 A arm，避免静默影响 C。

### 锁定匹配方案 v1

记同层已占用码的 A 均值表为 M（先 PCA 中心化逐物品归一化再分组）；P 为同一个固定 PCA 基，C 为该层残差质心。将 V=C@P 零填充至模型维度，在已占用码上计算每个坐标的均值。

```text
Mc = M - mean_rows(M)
Vc = V - mean_rows(V)
R = Vc * std_all(Mc) / std_all(Vc) + mean_rows(M)
E_A = M * std_all(random_table) / std_all(M)
E_R = R * std_all(random_table) / std_all(M)
```

std 均为 population std。残差表不逐行 L2；不从残差质心减完整物品 PCA 均值（两者原点不等价）。该方案匹配 A 的逐坐标均值与全表中心化能量，保留残差码相对几何。残差含A均值这一共同平移，故结论严格限于“匹配一阶/总尺度后的深层码几何方案比较”。仍不能排除逐物品非线性预处理、各码范数及存储质心的有限训练误差；不称为纯语义因果隔离。

退化残差投影或A中心化能量为零必须报错。新模式不新增训练参数、不消耗全局随机流。checkpoint 额外保存模式、映射版本及码本 tensor hash；默认A保持历史契约完全兼容。加载时拒绝不匹配模式/码本。

### 来源与装配

data helper 经共享 `resolve_reference` 解析 `quantizer_checkpoint_path`，校验文件 SHA256、层数、码表形状、初始化状态与有限值，不实例化量化器。配置记录明确 run/file/role，沿用 lineage callback；版本号不是当前解析器的alias，不用alias=v3伪装版本选择。模型检查与 catalog 维度一致，并固定局部seed抽查最多256个既有SID的逐层最近质心身份。检查不参与训练随机流、不读取交互标签。

### 判读顺序与停止条件

主指标 val/NDCG@10、辅助 Recall@10，采用既有 best-val checkpoint 规则；同时检查训练曲线、同一步19000和最终共同可用步骤，防止只挑最佳点。先核对完整20k预算、seed42、GPU0/1每卡batch128、lr0.0005、每500step验证及数据/产物完全一致。

- A 不优于 first_only：尚无深层必要性证据，优先简化；单seed小差异不能宣称等价。
- A 优于 first_only，但不优于 deep_residual：支持深层语义初始化，不支持完整内容代表量独有优势。
- A 同时超过两者：仅为候选机制的单seed初步支持；下一阶段才考虑锁定checkpoint的同用户推理、配对prefix bootstrap及独立seed复核，不自动追加。
- 曲线/最佳点冲突或差距小：记为不确定，不据此引入新模块。两个对照不能估计完整内容与残差预处理各因素的独立因果效应。

不预设成功阈值或显著性，不用静态几何指标替代推荐收益。单seed验证集选择结果不能支撑完整论文结论。

## Risks / Trade-offs

- 复用历史A节省训练，但环境可能漂移 → 训练后核对 resolved config 和环境，仅在发现实际失配时补基准。
- 额外下载量化器只是新条件启动依赖 → 精确文件和SHA256校验，避免best/last歧义；源文件可用本地路径替代，内容hash不变。
- 两个条件合计40k训练steps → 默认串行占用物理GPU0/1，由用户手动启动；也支持单条件分别指定`--gpus 0,1`/`--gpus 2,3`及不同`--master-port`并行。CUDA_VISIBLE_DEVICES选择物理GPU，Trainer始终使用逻辑devices=[0,1]；每任务仍为两个进程。可选单条件接续，不自动重跑已完成条件。

## Migration Plan

新增组件配置和根启动脚本，保持现有入口。单测/配置/脚本检查及OpenSpec校验后，Mutagen flush并确认四个session Watching且无冲突。撤销使用仅需回到原A配置；不更改已有checkpoint。

## 冻结推理扩展

两份推理组件先继承对应训练初始化组件，再继承标准catalog inference组件，保留token_initialization和残差码本来源，启用同用户prefix trace，metrics=null。沿用tiger_catalog_grounded_inference.sh，第一条件46kuqs6m固定19500步，第二条件nq8he993固定19000步。每项单卡、不同master port、不同task_name；evaluation与beam10保持与历史A一致。A的推理复用须核对输入checkpoint digest与输出身份，完成后仍需用户key/label对齐。该evaluation参与过checkpoint选择，不能写成独立测试集结果。

## Open Questions

## 固定实现的seed复核

用户已授权Beauty seed43、44各A/残差一对，共四次训练。新增`paired`入口按seed顺序执行full_content和deep_residual，A直接使用原组件并显式full_content。无模型实现变更，bank_seed仍42；两终端可使用不同GPU对和端口并行，详见seed-replication-plan.md。保留旧both行为；保存核心实现与配置SHA256以便事后审查。

实现口径已锁定；单seed训练验证已支持优势，配对推理结果与跨seed稳定性仍待确认。
