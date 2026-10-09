# CoPMRec 排序 loss 权重有界验证

## 问题与固定决策

用户于2026-10-04授权验证“loss等权可能是当前性能差的重要原因”。上一轮 f91njtjx 的6批梯度及候选覆盖诊断支持优化失衡风险，但未验证实际优化后的推荐收益。本阶段只鉴别排序权重，不新增模型组件，不冻结任何推荐参数。

固定预算为零更新校准加两臂各1000更新，共2000更新；不自动追加第三臂或完整训练。校准完全使用training数据，不按validation指标挑权重。若梯度改善且基础检索/自然覆盖恢复，支持重新加权；仅梯度改善则保留推荐收益不确定；无恢复或负向则收缩“等权是主要原因”的主张。

## 输入与协议

- parent：f91njtjx，原validation最优 `checkpoint_epoch=000_step=017500.ckpt`，已独立复制到node1本地诊断目录。
- SHA256：`5107b13b9fa35cec5d9a19ccaef4f85ee54afdd490025f313bcbae8c37f96e62`。
- SID：dq77e3wo；内容embedding：3jtt9mpa，使用与原run一致的本地缓存字节。
- 两臂恢复同一模型、AdamW和scheduler，scheduler总长50000、起始step17500，固定终点18500。
- GPU1单卡，batch128、累积2、每微批排序4例、FP32、梯度裁剪1，seed42、training worker0/timeout0；两臂均用现有 `src.main experiment=copmrec_v1_1_train`。
- 与父run双卡分片执行不同，不能声称精确复现父run轨迹；两臂间仅权重变化。
- 短程训练不运行完整validation；固定终点由共享ModelCheckpoint保存，再做同一2048用户selection诊断。固定终点是预先声明的匹配诊断，不冒充正式best或audit/test。
- W&B group：`copmrec_loss_weight_verify_20261004`，job_type：diagnosis；真实runtime源码由统一入口归档。

## 零更新校准结果

32对真实training微批，前16对校准，后16对保留检查。每对先平均两微批梯度，再测基础三项目标合计与raw ranking梯度；实际dropout开启。

校准原梯度比中位数8.7309，目标取0.5，因此唯一续训权重为 `0.05726763550972437`。

| 保留training样本，中位数 | 权重1 | 校准权重0.05727 |
|---|---:|---:|
| 加权ranking/基础梯度范数 | 6.4618 | 0.3701 |
| 共享主干ranking/基础梯度范数 | 4.4477 | 0.2547 |
| 合成梯度与基础梯度余弦 | 0.1870 | 0.9371 |

这确认固定checkpoint处的优化方向可被权重显著改变，尚未单独证明AdamW参数位移或推荐收益。其他权重仅为同一梯度的零更新尺度参考，没有启动相应训练。

## 两臂短程结果

两臂均有效完成1000次optimizer更新。等权run为 [68h3e3eq](https://wandb.ai/baymaxam/GRID/runs/68h3e3eq)，校准权重run为 [mg0nrz9a](https://wandb.ai/baymaxam/GRID/runs/mg0nrz9a)。下表来自固定终点的独立函数级评价，保存在本地证据；不是这两个W&B run的完整validation指标。

| 固定2048个selection用户 | parent 17.5k | 等权续训1000步 | 校准权重续训1000步 |
|---|---:|---:|---:|
| content CE | 8.88112 | 8.83905 | 8.77442 |
| SID CE | 4.15268 | 4.11573 | 4.06196 |
| dense Recall@10 | 0.008301 | 0.008789 | 0.014648 |
| dense NDCG@10 | 0.003982 | 0.004726 | 0.006495 |
| 自然候选覆盖率 | 0.018066 | 0.015625 | 0.021973 |
| 候选内仅内容 Recall@10 | 0.008301 | 0.009766 | 0.012207 |
| 候选内仅内容 NDCG@10 | 0.004628 | 0.005842 | 0.005647 |
| 最终hybrid Recall@10 | 0.009277 | 0.006348 | 0.012695 |
| 最终hybrid NDCG@10 | 0.004314 | 0.003617 | 0.006439 |
| 自然候选覆盖目标数 | 37 | 32 | 45 |
| 最终Top10命中数 | 19 | 13 | 26 |

校准权重相对等权：dense新增21/丢9、净增12命中；最终hybrid新增21/丢8、净增13命中。dense Recall差的配对95%区间为 `[0.0009766, 0.0112305]`，最终hybrid Recall差为 `[0.0014648, 0.0112305]`。最终hybrid NDCG点增0.0028222，区间 `[-0.0000391, 0.0059633]` 仍跨零。区间为固定checkpoint/样本条件下2000次未校正的逐用户配对bootstrap，不是跨seed或完整训练置信区间。

仅候选内内容NDCG略降，说明并非所有路径/指标一致改善；最终推荐收益不能用基础CE或单一路径代替。

## 判断与剩余不确定性

验证支持“等权是当前v1.1退化的重要因素”：权重降低同时改善保留training样本的优化方向、基础CE、dense命中、自然覆盖和最终hybrid命中，证据已超出单纯梯度尺度。当前固定实例应保留经training校准的低权重候选，不优先新增其他模块。

尚未证明该权重是全程最优、完整从零训练收益、跨seed泛化或相对v0/LIGER的最终优势；不能量化它解释了多少历史差距。parent和AdamW状态已受17.5k等权训练影响，短程续训只能回答恢复方向。固定阶段已完成2/2臂、2000/2000更新，不自动追加权重扫描、第三臂、audit或完整训练。

## 实现与验证

`ranking_loss_weight`有限正数、默认1；raw ranking日志保留。恢复时默认拒绝权重变化，只有 `allow_ranking_loss_reweighting=true` 允许仅权重字段变化并记录来源。其他版本/目录/评分/历史契约仍严格检查。

2026-10-04后续采纳：用户要求采用校准权重并从零训练。v1.1组件配置默认值更新为0.05726763550972437，Python API与v1组件默认1保留；新的50k从零命令显式设置 `ckpt_path=null`、`pretrained_checkpoint_path=null` 和禁止恢复重新加权。完整命令见 [v1.1训练命令](copmrec-v1-1.md)，未自动启动；已完成的短程验证仍保持原2/2臂和2000更新证据，不追改其配置或结果。

67项聚焦测试通过，Ruff、Hydra compose和OpenSpec strict通过；Mutagen三会话Watching无conflict。本阶段未启动新的完整训练、未停止原run。

两组的model/data/trainer配置除权重及独立output目录外一致，实际runtime source SHA256均为 `da7f6113e0ab09e9eb99179bfef148df13af55640b852d2f7c6d9b2079a47564`，origin verified。相关执行文件与两组source archive字节一致。两组全部157个参数张量保留requires_grad，并由同一个optimizer覆盖；所有AdamW参数step和scheduler last_epoch均为18500，总schedule仍为50000。选择用户keys SHA256为 `46ab08876eee752143cb91ce45024eee9c6fb5c3bbddeda466ad44f87ac7776a`，完整评价history/selection契约与parent一致，输出Top10目录合法且无重复。

固定终点checkpoint SHA256：等权 `c7b00dc9f7cbd21fcf348680acae9318bbabb1b9f27c1f06054744230476e195`；校准权重 `3e08284da403df53f62c33cdb5e7eba6b03782f1b5a5212204d9ce043427e05e`。读取last仅用于预先声明的固定终点诊断，不作为正式validation-best选择依据。

## 证据

- [training梯度校准](evidence/copmrec-loss-weight-20261004/calibration.json)
- [固定parent原始脚本](evidence/copmrec-loss-weight-20261004/pin-parent-source.txt)
- [校准原始脚本](evidence/copmrec-loss-weight-20261004/calibration-source.txt)
- [统一入口续训命令源](evidence/copmrec-loss-weight-20261004/continuation-launch-source.txt)
- [实际运行命令与退出状态](evidence/copmrec-loss-weight-20261004/launch.json)
- [W&B配置、源码指纹与有界训练历史](evidence/copmrec-loss-weight-20261004/wandb-evidence.json)
- [逐用户终点评价及配对结果](evidence/copmrec-loss-weight-20261004/evaluation.json)
- [终点评价原始脚本](evidence/copmrec-loss-weight-20261004/evaluation-source.txt)
