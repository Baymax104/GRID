# 第二层内容交互：seed42 首项训练结果

核验日期：2026-09-18。来源为 W&B `baymaxam/GRID` 的在线 config、summary、完整40个validation点、checkpoint metadata和输入Artifact lineage。此次只读分析，未执行训练、推理或testing。

## 当前结论

interaction 尚未通过预先固定的晋级门槛：best-validation NDCG 相对 A 为 -0.6374%，同点 Recall 为 -0.5583%；后五点 NDCG 均值 +0.5579%。不能用后五点或最终步的局部改善替换 best-validation 主判据。

按协议筛选只找到 interaction 一项，additive/shuffled 尚未查到；三条件第一阶段尚未完成，不能得出交互优于加性或置乱的机制结论。按原计划补齐两个预先锁定的seed42对照，仅用于完成归因；当前不进入seed43、不自动调门学习率或尺度、不启动testing。单seed负信号不证明所有前缀交互设计无效，也不构成显著性结论。

## 成对指标

| 指标 | 原A，5g3wpbg7 | interaction，8rgf0c7l | 相对变化 |
|---|---:|---:|---:|
| best-validation NDCG@10 | 0.04256973 | 0.04229838 | -0.6374% |
| 同一最佳点 Recall@10 | 0.08003333 | 0.07958652 | -0.5583% |
| 最后五个validation点平均 NDCG@10 | 0.04157764 | 0.04180958 | +0.5579% |
| 最佳checkpoint步数 | 19000 | 19500 | — |
| 最终步 NDCG@10 | 0.04173620 | 0.04199509 | 仅作辅助观察 |
| W&B run runtime，秒 | 1878 | 2401 | +27.85% |

两项均finished、seed42、20k步、40个验证点。日志的trainer/global_step为零起点，最佳点18999/19499分别对应文件中的19000/19500。summary的validation值是最终点，并非最佳点，本表由完整曲线选出并与checkpoint metadata核对。

## 契约与产物

- 原A：[5g3wpbg7](https://wandb.ai/baymaxam/GRID/runs/5g3wpbg7)。候选：[8rgf0c7l](https://wandb.ai/baymaxam/GRID/runs/8rgf0c7l)。
- 两项共同使用 `rkmeans_inference-semantic-id:v2`，digest `20f08b323a286fbb3f16b5ea27562af1`；`sem_embeds_inference-semantic-embedding:v5`，digest `ab56af975eac589c27eed6094482cbd7`；Artifact ID一致。
- 配置差异为新增分支、训练监测、实验元信息/输出路径和等价的显式Artifact URI。训练预算、优化器、batch、精度、验证协议配置未变；历史实际源码、数据文件hash和完整硬件/软件环境尚未独立核实，不称严格同环境对照。
- 候选最佳产物：`tiger_level2_interaction_beauty_train-checkpoint:v0`，文件 `checkpoint_epoch=000_step=019500.ckpt`，88,102,229字节，文件manifest digest `2J5V5zepgHnMgqpHxBp38g==`；metadata selection=best、monitor=val/ndcg@10、score=0.04229838401079178。
- 此次未下载或恢复checkpoint，内部固定表及尺度hash尚未作加载复核。

## 机制与成本观察

最后一个训练日志窗口的 g≈0.8302、tanh(g)≈0.6806、注入/基础embedding RMS比≈0.8178。指标表明分支实际参与了训练；窗口汇总值不等于最佳checkpoint的精确门值，也不能证明因果收益或门过强。

统一取global_step 2000–18000，排除与validation日志相邻的间隔，以相邻训练日志的step差/runtime差估计：每项288个间隔，中位数A≈14.42 step/s、候选≈12.58 step/s，候选约下降12.76%。该值为训练日志区间估计，排除了validation，不能与包含验证的历史总体step/s混用。不是同机交错控制benchmark；硬件并发负载差异未排除。

总runtime增加27.85%，与训练区间约12.76%的吞吐下降不是同一口径。已触及预设10%成本调查线；若继续投入，应先用受控性能测量区分训练、验证和同步成本，不能直接归因于某一段实现，也不降低验证频率规避成本。

## 后续手动操作

保留这次结果及checkpoint，完成第一阶段余下两个对照；全部条件结果完整报告。目前不建议运行seed43。

```bash
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition additive --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; seed42; additive control; complete preregistered first stage"
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition shuffled --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; seed42; shuffled control; complete preregistered first stage"
```

必须从GRID根目录运行，确认指定GPU可用；每条独立手动执行。完整预算和门槛见[experiment-plan.md](experiment-plan.md)。

原始在线快照：`tmp/level2_results/initial-snapshot.json`、`tmp/level2_results/runs.json`；计算及完整配置差异：`tmp/level2_results/analysis.json`。候选检索范围为协议字段 `interaction_protocol=level2-content-interaction-v1` 或已知run ID；“未查到对照”限于此范围。
