# SID初始化差异的训练结果

## 结论

本轮支持继续以A为主方法候选：Beauty seed42、20k预算下，完整内容均值初始化同时优于仅首层初始化和均值/整体尺度匹配的深层残差质心初始化。优势不仅存在于各自最佳点，在训练后段和最后一步也存在。证据等级为单seed训练验证支持，尚非独立测试或跨seed机制确认。

## 完整性与可比性

2026-09-17通过只读W&B API重新采集既有A及两项新训练的config、summary、完整scan_history、used/logged artifacts。三项均finished，各40次验证，对应500至20000每500步；W&B历史global_step为零起始，本文训练步数按其加1，与checkpoint文件步数一致。

对model、data、trainer、callbacks以及seed/devices等协议字段逐项比较，并用实际artifact身份确认短URI与完整URI等价。剩余差异仅为预定初始化模式、残差码本配置和任务/输出路径；没有发现预算、模型规模、优化器、batch或验证规则的配置变化。三项均seed42，每任务逻辑devices=[0,1]、每卡batch128、lr0.0005、20k steps、每500step验证、ckpt_path=null。

注意：实际新run记录的物理GPU为first_only=[4,5]、deep_residual=[6,7]，不是之前建议的[0,1]/[2,3]。每项仍为双卡配置；仅凭卡号变化不否定可比性，但本轮未核验不同卡的硬件型号、驱动和完整环境一致性，也未重新哈希远端交互数据文件。

三项共用SID `rkmeans_inference-semantic-id:v2`（digest `20f08b323a286fbb3f16b5ea27562af1`）及内容 `sem_embeds_inference-semantic-embedding:v5`（digest `ab56af975eac589c27eed6094482cbd7`）。残差run额外使用正确的 `rkmeans_train-checkpoint:v3`（digest `312fbcb197a710aaa6f40f36d1e0a08f`），配置中的文件SHA256与既定来源一致。

## 指标

下表Recall为最佳NDCG checkpoint对应值，不是单独最大Recall。

| 条件 / run | 最佳NDCG@10 | 对应Recall@10 | 最佳步数 | 19000步NDCG | 20000步NDCG | 最后5次验证NDCG均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / 5g3wpbg7 | 0.04256973 | 0.08003333 | 19000 | 0.04256973 | 0.04173620 | 0.04157764 |
| first_only / 46kuqs6m | 0.03769524 | 0.07198055 | 19500 | 0.03674737 | 0.03669138 | 0.03646729 |
| deep_residual / nq8he993 | 0.04061059 | 0.07680967 | 19000 | 0.04061059 | 0.04043356 | 0.03986626 |

A相对first_only的最佳NDCG提升12.93%，绝对差0.00487449；相对deep_residual提升4.82%，绝对差0.00195914。

A在40个共同验证点中，分别37次超过first_only、30次超过deep_residual；最后10次验证均超过两者。这是曲线一致性证据，不把相邻checkpoint视作独立重复实验，不据此计算显著性。

## 已固定的checkpoint

| 条件 | 输出artifact | 文件 | digest |
| --- | --- | --- | --- |
| A | tiger_catalog_grounded_token_content_init_beauty_train-checkpoint:v0 | checkpoint_epoch=000_step=019000.ckpt | aaba5b47dc9a5e94c0d23b75cf697238 |
| first_only | tiger_content_initialization_first_only_beauty_train-checkpoint:v0 | checkpoint_epoch=000_step=019500.ckpt | 4f2fe5145446eee3bbdeb6089b00f043 |
| deep_residual | tiger_content_initialization_deep_residual_beauty_train-checkpoint:v0 | checkpoint_epoch=000_step=019000.ckpt | a5fad97f3a5ae19ce8adb9d49de15457 |

本轮只核查W&B artifact元数据与已发布文件列表，没有下载新训练checkpoint或启动推理。

## 机制含义与下一步

1. 首层不足以解释当前A的全部效果，深层初始化具有实际贡献；本轮不能拆分第2层、第3层各自的独立贡献。
2. deep_residual高于first_only，而A进一步更好，支持“深层代表量的选择有用”，并非任何深层初始化都同样有效。
3. 可以将主线收敛为：量化残差代表量未必最适合生成token初始化，使用完整物品内容的条件均值能提供更有效的深层初始化。这里“更有效”限定当前数据、seed和匹配方案；不声称残差码一定丢失有用语义或证明通用因果机制。
4. 不恢复C/D/E，不加新模块，不马上扫参数。下一步优先冻结上述三份checkpoint，补两个新条件的同用户推理，复用并核对已有A推理，做配对prefix bootstrap。训练seed不确定性需要后续独立seed验证，bootstrap不能替代；只有这一步仍支持优势时再限定复核预算。

## 来源与复算

- [A](https://wandb.ai/baymaxam/GRID/runs/5g3wpbg7)
- [first_only](https://wandb.ai/baymaxam/GRID/runs/46kuqs6m)
- [deep_residual](https://wandb.ai/baymaxam/GRID/runs/nq8he993)
- 本地原始快照：`tmp/sid_initialization_results/runs.json`。
- 统计与协议差异：`tmp/sid_initialization_results/analysis.json`。
- 曲线：`tmp/sid_initialization_results/validation_curves.png`。
- 只读采集：`uv run python -m tmp.inspect_sid_initialization_results`；本地复算：`uv run python -m tmp.analyze_sid_initialization_results`。

未修改生产代码、未同步代码、未创建Git提交、未发布新run、未启动训练或完整推理。
