# v3.1 固定 alpha=0.813：内容遗漏减少，整体收益未确认

日期：2026-10-04。用户单卡推理 run [3lb22b0r](https://wandb.ai/baymaxam/GRID/runs/3lb22b0r) 已 finished；独立核验完整22363名Beauty Testing用户、实际输入、checkpoint、预测/trace及运行源码。对照为同一best43500、学习alpha=0.668786的v3.1 [zznw291t](https://wandb.ai/baymaxam/GRID/runs/zznw291t)。固定0.813使dense Top10遗漏81→52，同时超dense增量命中81→52；Recall不变，NDCG10点估计+0.0775%、用户配对区间跨零，未建立整体净收益。保留原v3.1学习权重默认设置，不晋级固定0.813，不自动扫描。

## 对照身份与有效性

- resolved model config相对zznw291t仅 `inference_mixture_alpha: null→0.813`；同v3.1 RootEnvelopeCoPMRec、seed42、FP32、logical[0]、predict batch32、beam20与content Top10、all cold33。
- 实际消费 `copmrec_beauty_v3_train-checkpoint:v0`，producer rbha00vx，digest `48204b0d67bf2ba6c244d6f1b09d583a`，best43500，文件MD5/base64 `hO6s8wKBi1kK3MD2hmR6OQ==`；best/val-NDCG/max元数据与v3训练/逐层聚合契约一致。
- candidate/path metadata实际alpha=0.813、source=fixed_inference、checkpoint alpha=0.668786346912384。所有可观测目标局部混合概率均与固定公式一致，最大log概率误差9.54e−7；root包络一次补正及累计搜索等式通过专用validator。
- 同一SID与embedding producer/digest，catalog12101、cold33，checkpoint catalog buffers与预处理一致。全部用户/真实目标来自原始Testing TFRecord独立构造，预测bundle shape[22363,10,4]、合法且无重复，并与candidate/path末层一致；独立Recall/NDCG及candidate summary匹配W&B。
- 与zznw291t dense Top10及目标dense rank逐项相同；共享父前缀生成条件log概率差0、内容差≤9.54e−7。首层Max规则保留但root集合允许改变：17263/22363用户首层成员发生变化，平均新增1.136个root。不得套用上一轮相同alpha的frontier不变门禁。
- 实际code Artifact `grid-source-3lb22b0r:v0` 的3文件大小/MD5、manifest/archive SHA256通过，归档31个相关运行文件与交付字节一致，source_sha256=`f8ab9b23b26e99361e0e03b00fdf9b1673c906941c255e5481d0311c8a0d3172`。本轮不修改运行代码或启动模型前向/decoder search、新训练/推理。

## 最终指标与区间

| 方法 | Recall5 | NDCG5 | Recall10 | NDCG10 |
|---|---:|---:|---:|---:|
| v3.1 学习alpha0.668786 | 0.045432187 | 0.027806847 | 0.070384117 | 0.035827804 |
| v3.1 固定alpha0.813 | 0.045432187 | 0.027830691 | 0.070384117 | 0.035855580 |
| 真实v0最终 | 0.043822385 | 0.027383122 | 0.071904485 | 0.036448314 |

固定减学习的Recall10差值0、95%区间[−0.000760184,+0.000804901]；NDCG10差值+0.0000277757、相对+0.077525%、区间[−0.000237653,+0.000308928]。用户配对bootstrap2000、NumPy PCG64 seed42、未校正逐项区间，条件于固定checkpoint；不是多seed训练确认，区间跨零不证明等价。

固定alpha相对真实v0 Recall10−2.114%、NDCG10−1.626%；相对LIGER dense分别−1.440%/−1.471%，相应配对区间均跨零。LIGER对照来自全目录dense trace，不使用其hybrid summary。

## 命中与候选交换

| 同一dense评分下的统计 | 学习alpha | 固定alpha |
|---|---:|---:|
| 高内容目标未入候选 | 81 | 52 |
| 分层遗漏 | 0/60/11/10 | 0/44/6/2 |
| 超出dense Top10的最终增量命中 | 81 | 52 |
| 全体目标候选覆盖 | 2471 | 2493 |

原81个遗漏恢复36个到候选和Top10，同时新增7个高内容遗漏，净恢复29个。原81个超dense增量保留48个、损失33个，新增4个增量，净损失29个。于是：

```text
新增命中 = 36个高内容恢复 + 4个新生成侧增量 = 40
丢失命中 =  7个高内容遗漏 +33个旧生成侧增量 = 40
最终净命中 = 0
```

候选覆盖新增154、损失132、净增22，仍未形成最终Recall收益。原v3首层恢复47个候选全部保留，原29个Top10命中保留28个；原v3.1在v0首层失败群体的31个命中保留30个。

33个丢失的增量中，仅5个目标从候选消失，28个仍在候选但跌出Top10，其中25例伴随新dense Top10候选进入。固定content终排使更高内容商品直接竞争位置，不能把这28例记为候选漏召回。本轮24个“单商品目标对多商品兄弟”结构案例恢复7个、仍漏17个；仅为已查看Testing上的描述，不证明候选完成上界的效果。

## NDCG账目与解释

共同Top10命中12人升位、92人降位（其中81个目标本身属于dense Top10）。精确分解为：

```text
新增命中贡献       +0.00066208268
丢失命中贡献       −0.00053042736
共同命中位置变化   −0.00010387965
合计               +0.00002777567
```

提高内容权重在同一模型下确实保护了更多强内容目标；同时减少了内容终排下生成侧能保留的增量，并使一部分共同命中降位。支持“候选选择与content终排越一致，越倾向于保护高内容候选”的局部解释，但不能确认content终排是高alpha偏好的唯一原因：本次未改变终排，也未实施alpha×终排的交互对照。不能将0.813称为最优权重，不能用跨checkpoint的v0优于v3解释为纯alpha效应。

## 阶段决定

该有效对照缩小了“推理权重差异能否补齐v3.1推荐收益”的不确定性：提高alpha减少遗漏，但净命中被增量损失完全抵消，NDCG仅有区间跨零的微小点估计，未通过整体收益门槛。原v3.1和首层Max保留，默认继续checkpoint学习alpha；固定0.813留作机制证据，不晋级、不自动追加扫描/重训。

用户授权的一次完整推理已消费，剩余额度0；本轮分析新运行0、训练预算0、W&B/Linear写入0。此前完成上界方案仍未实施、未获运行授权，原关闭阶段预算保持。Testing已用于开发分析，结论不包装为独立确认。

证据：[运行快照](evidence/copmrec-v3-1-alpha0813-testing-3lb22b0r-20261004/wandb-run.json)、[完整复算](evidence/copmrec-v3-1-alpha0813-testing-3lb22b0r-20261004/results.json)、[增量损失与首层保留](evidence/copmrec-v3-1-alpha0813-testing-3lb22b0r-20261004/focused-analysis.json)、[核验记录](evidence/copmrec-v3-1-alpha0813-testing-3lb22b0r-20261004/verification.json)。
