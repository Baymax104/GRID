# CoPMRec v2 M3 启动与已有来源机制分析回执

日期：2026-10-09。BMX-157当前In Progress：三臂训练已在node1独立tmux启动；已有Full来源的残差与前缀分析直接执行并核验完成。其余分析等待实际来源，不标完成。Full复用BMX-120已有效的rmgfscr0 / 57jvuy59。

## 训练启动

| Issue | Variant | Run | tmux | 物理GPU → local | 端口 |
| --- | --- | --- | --- | --- | --- |
| BMX-158 | no_mixture | [qhzzhnxp](https://wandb.ai/baymaxam/GRID/runs/qhzzhnxp) | `bmx-158-v2-train-qhzzhnxp` | [0, 1] → [0,1] | 29860 |
| BMX-159 | no_residual | [gtwv2c4n](https://wandb.ai/baymaxam/GRID/runs/gtwv2c4n) | `bmx-159-v2-train-gtwv2c4n` | [2, 4] → [0,1] | 29861 |
| BMX-160 | no_joint_ce | [izwmwyig](https://wandb.ai/baymaxam/GRID/runs/izwmwyig) | `bmx-160-v2-train-izwmwyig` | [5, 7] → [0,1] | 29862 |

三臂独立scratch，每臂50k、DDP2/global256/FP32。实际W&B ablation配置、有限loss、非零optimizer步数、两个GPU进程与runtime source已核验；启动不表示训练完成或正式指标有效。Testing须待本臂首次raw val/ndcg@10最高own-best产生并核验。

## BMX-162：固定Full残差2×2实测

h为历史残差、c为目录残差，Vhc四条件；全部22363名Testing用户。以下仅提供原值及带符号差，推理干预不等同于重训A2。

| 视图 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | NDCG@10差 | 相对变化 | 用户配对95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v11 | 0.04807047 | 0.03194638 | 0.07208335 | 0.03967084 | +0.00000000 | +0.00% | [+0.00000000, +0.00000000] |
| v10 | 0.00818316 | 0.00533872 | 0.01498010 | 0.00751534 | -0.03215550 | -81.06% | [-0.03432497, -0.03001523] |
| v01 | 0.02106157 | 0.01356302 | 0.03599696 | 0.01836701 | -0.02130383 | -53.70% | [-0.02357012, -0.01911050] |
| v00 | 0.00630506 | 0.00379954 | 0.01265483 | 0.00583676 | -0.03383408 | -85.29% | [-0.03595951, -0.03170463] |

数值交互 I=m11−m10−m01+m00：recall@5=+0.02513080, ndcg@5=+0.01684419, recall@10=+0.03376112, ndcg@10=+0.01962525。

V11逐key完全复现现有Full Testing Top10；四bundle均为完整合法唯一Top10。原始TFRecord用户/标签、target seen/cold、training-only频次与历史长度重新对齐；固定query的cold logit在目录残差开关间逐字一致。四指标、切片均值、全部非空切片的2000次PCG64 seed42 pointwise配对CI已独立复算。区间不表示训练seed不确定性。

## BMX-163：Full真实前缀实测

Full run `2nosdem5`：22363名用户×4层=89452条记录，global alpha=0.9887975824172957，按训练定义的全量mean mixed NLL=2.317011970980427。A1尚无own-best，跨来源差异保持null。

| Depth | N | p_gen(target)均值 | p_mass(target)均值 | p_mix(target)均值 | gen NLL均值 | mass NLL均值 | mixed NLL均值 | JS均值 | argmax一致率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22363 | 0.04113611 | 0.04506736 | 0.04502331 | 4.73993646 | 4.79621538 | 4.79199835 | 0.05289186 | 0.48200152 |
| 2 | 22363 | 0.13646629 | 0.15612274 | 0.15590254 | 3.25492695 | 3.53809279 | 3.47849299 | 0.19473828 | 0.35674999 |
| 3 | 22363 | 0.71833760 | 0.72267494 | 0.72262635 | 0.94807357 | 0.80130391 | 0.78295320 | 0.07952444 | 0.77274963 |
| 4 | 22363 | 0.91041449 | 0.91404007 | 0.91399946 | 0.23854883 | 0.22137336 | 0.21460334 | 0.02285765 | 0.92442874 |

独立核验目标前缀合法、概率/entropy/rank范围、p_mix=(1−alpha)p_gen+alpha p_mass、NLL与目标概率、连续差及固定bin，以及所有切片均值和10/50/90%分位。完整合法支持归一化由正式诊断运行时检查；独立审计复算保存的目标概率与统计，没有再启动模型前向。全局alpha不解释为用户级差异，也不推断beam恢复率或dense效果的因果中介。

## 当前依赖与来源

BMX-161待三臂训练/own-best/单卡Testing后，使用四份当前输出做零forward列表分解；BMX-163 A1前缀待BMX-158 own-best。训练启动后不持续监控，也没有安排依赖分析消费尚不存在的Artifact。

源码SHA256 `3e486ce5d7a206c13bee21f108d1e806f862712a3cc1afd5a6535bcd06bd8a00`；Full checkpoint SHA256 `d2a70c53272116b687cf62592aa0823bc42f23cb40c7ca1c8dcd4012ce0060bf`；精确URI及实际命令见[启动计划](launch-plan.json)。运行前Mutagen flush成功，三个session Watching/no conflict；源码与origin核验、Beauty三split逐文件SHA及Full checkpoint SHA通过。

两项诊断直接运行，均单卡GPU7 / no optimizer updates / finished / exit0。source archive、Artifact manifest MD5、冻结输入digest和checkpoint身份已核验。完整结果保存在node1 `logs/copmrec_v2/bmx-162_residual_full_m03ltj2f/analysis` 与 `logs/copmrec_v2/bmx-163_prefix_full_2nosdem5/analysis`；[独立审计](diagnosis-audit.json)。

W&B runtime登记；峰值显存和active GPU hours未单独采集，保持null。只报告实证，不按数值方向判定完成。
