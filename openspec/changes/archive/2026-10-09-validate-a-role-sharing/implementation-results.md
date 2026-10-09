# A角色共享诊断：实现与手动运行

日期：2026-09-18。用户已完成正式诊断28w08e28；技术检查通过，但效果门槛未通过，建议暂停结构推进。见[正式结果与结论边界](results-28w08e28.md)。下文保留实现和运行协议，运行前状态以本段最新结果为准。

同步状态：用户恢复node1连接后，已成功执行当时路径下的 Mutagen flush；status确认当时四个session均为Watching for changes，双方端点连接正常且无conflict。此前SSH超时阻塞已解除，可手动运行下方命令。当前管理入口为 `./mutagen_sync.ps1`。

## 冻结条件

- Beauty / RKMeans / A full_content / seed42。
- checkpoint：`wandb://baymaxam/GRID/5g3wpbg7?role=checkpoint&file=checkpoint_epoch=000_step=019000.ckpt`。
- SID：`wandb://baymaxam/GRID/4vyi4o6w?role=semantic_id&file=merged_predictions_tensor.pt`。
- embedding：`wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt`。
- 32名training末目标用户生成平均梯度方向；64名不同用户的evaluation末目标用于检验；固定hash seed 20260918。不使用testing、不随机扩增窗口、不执行优化器。
- FP32 highest、单进程、batch1、inference_mode=false。只有临时embedding张量需要梯度，原模型所有参数冻结。

## 比较的13个条件

baseline + 两个预定相对联合范数幅度0.001/0.003，各含：shared、split、history、prefix、antisymmetric、random_antisymmetric。

每个非零方向按两张角色表的联合范数匹配。反对称方向对应共享约束之外的自由度；随机反对称方向逐token匹配幅度，避免单纯活跃token分布造成差别。全目录精确概率验证归一化，记录目标CE、精确rank及容差区间、TopK。

共享更新与拆分更新匹配的是实际扰动量，不是相同学习率；这是局部敏感性比较，不是训练配方优劣比较。零方向会显式记录为零，不能当作有效处理。

## 手动命令

从node1仓库根目录执行。物理GPU默认0，可自行选择空闲卡。

```bash
bash ./tiger_role_sharing_audit.sh \
  --data-dir /data3/weizhenyu/projects/GRID/data/beauty \
  --gpu 0 \
  --notes "A role-sharing v1: disjoint train support and evaluation; fixed two radii; no training"
```

可先在同一命令中添加`--dry-run`：仅2名支持用户、1名评价用户及baseline/shared/split首幅度；统一launcher禁止结果发布。这仍会加载模型，因此由用户手动执行。正式运行不带该参数。

数据目录应包含training与evaluation子目录；若远端实际数据根不同，只更改`--data-dir`。`--seed`不会自动更换固定checkpoint，不应仅改seed就宣称完成另一个训练seed的复核。

## 证据和后续判断

输出：`${paths.output_dir}/audit_evidence/role_sharing_audit.pt`，并发布W&B Artifact，role/type为`role_sharing_audit`。包含checkpoint/catalog指纹、支持用户及逐token梯度统计、输入指纹、方向指纹和评价用户条件结果。Artifact页面元数据仅放摘要，完整大数组在文件中。没有新模型checkpoint。

此诊断沿用已有checkpoint评分audit的predict生命周期；不是Tail-SID diagnosis，不改analysis/test入口。

结果回来后按以下固定顺序分析：

1. 先验证身份、support/evaluation互斥、零干预等价、梯度分解误差、实际扰动范数及概率质量。失败则无效，不解释效果。
2. 对每个幅度报告split与shared的逐用户CE配对差、精确NDCG@10/Hit@10及排名变化；两个幅度同时报告，不能挑一个最优值。
3. 对比单角色及随机反对称控制。分层查看梯度冲突与实际曝光，不能从负余弦直接推断损害。接近排名同分边界的变化单独标记。
4. 只有两个幅度方向一致、对shared的配对CE改善有可靠区间支持、排名没有系统性代价，并优于随机控制时，才建议复核现有A seed43及匹配随机初始化checkpoint。bootstrap重采样单位是evaluation用户，属于开发证据。
5. 区间宽或正负不稳定应记“证据不足”，不宣称普遍不存在角色瓶颈；暂停结构立项。若普通解除共享足够，则无必要叠加新模块。
6. 只有跨checkpoint信号稳定，才另行设计同起点/同预算/同优化器状态的短续训。当前不启动该阶段，也不恢复G1、CCFD或互补。

## 验证

- 41项聚焦测试通过：角色与共享前向/梯度等价、有效曝光、异常hook清理、参数不变、范数与随机行幅度匹配、标签隔离、用户抽样、数据目录路由、writer roundtrip、Hydra装配、shell quoting/空值/错误参数/override透传，以及已有A审计回归。
- Ruff检查通过；`openspec validate validate-a-role-sharing --strict`通过。
- 实现验收阶段未运行完整Trainer实验；其后用户完成28w08e28，实际诊断结果见上方报告，本诊断不评估训练速度。
- Windows沙箱曾阻止临时目录和Bash测试；在获准的沙箱外范围重跑上述纯测试后通过，不涉及真实实验。
