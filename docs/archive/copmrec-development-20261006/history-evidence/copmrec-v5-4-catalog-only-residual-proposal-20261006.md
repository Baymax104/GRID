# 下一固定验证方案：仅目录侧协同残差（v5.4，尚未授权）

## 当前状态与完整目标

这是可审阅方案，没有实施模型、修改运行源码、启动新训练或生成新预测。当前主方案仍为v5.2；v5.3固定辅助CE已停止，其相对native的明确部分收益证据保留。本方案不恢复该CE、不复制第二query，也不改动已闭合阶段的门槛与费用。

完整目标保持：推荐模型随机初始化，全部模块共同连续训练50000更新，单模型／单checkpoint部署，相对同预算、同seed、同history资格的LIGER dense，R10与N10均至少+8%、两项paired绝对差CI下界为正，并完成既定配对复现及独立Testing。固定SID与内容Embedding仍是上游输入，不等于外部推荐checkpoint初始化。

已完成成本为4train／200000更新／4完整Validation，启动attempt5（其中预测前失败1次）、新Testing0、seed43 pair0。当前授权上限也是4train／200000／4Val，剩余额度为0；原账本、注册及闭合记录不修改。本方案只请求新增1train／50000＋1单卡完整Val，Test0、43 pair0、扫描0；若明确授权，累计才增加至5train／250000／5Val，不能在授权前登记reserved、started或committed。

## 先完成路线五问

1. **核心假设。** 内容与seen协同残差的整体联合模型能在50k内取得可复现双8收益。当前需要收敛的是seen用户覆盖恢复与既有命中损失的取舍；不再把cold占位作为主要瓶颈，也不主张某个因素解释全部损失。
2. **支持证据。** v5.2相对native R10+4.6587%、N10+12.9686%，两CI下界正，支持保留整体部分收益。v5.3相对native +7.0572%／+13.8794%，也支持整体部分收益；v5支持N10正向，R10仍不确定。所有四个候选都证实了随机起点、共同连续50k和单模型部署的可实施性。
3. **反证和未验证。** v5.1固定0.5／0.5双目录评分对v5的双指标CI为负，停止该固定评分。v5.3对v5.2的+2.2918%／+0.8063%增量CI均跨0，停止固定辅助CE，不能宣称已证明无效或等价。四个版本均未过双8；seed43配对复现与本scratch阶段的新Testing未运行。
4. **实现疑点。** 主审计、独立复核、来源与实际代码没有发现阻断当前取舍的实现错误。v5.3只移除辅助目录的residual，query仍含history residual，且CE不约束native原有商品顺序；它不是native排序保护。这是已实现干预的范围，不是需补修的漏项。原Top10无法分离共享history路径与目录路径的作用；原因允许保持未知，不为穷尽原因安排实验。
5. **原预算内是否追加实验。** 0个。新方案需要额外授权；长期目标、空闲GPU或这个proposal都不代表新增额度。

证据限定为当前保留的[v5.2结题](copmrec-v5-2-stage-decision-20261006.md)、[v5.3结题](copmrec-v5-3-stage-decision-20261006.md)及其绑定的实际审计；不以归档实验恢复停止路线。实际完整训练预算均为50k，但各自保存状态仅到41k／46k／47k／45k，不能把预算冒称50000终态完整checkpoint。

## 正面依据及决策价值

v5.2相对native有925个新增命中和824个丢失，二者存在实际seen覆盖互补。v5.3只监督去掉目录residual后的分数，对v5.2恢复B组149、新覆盖D组278，却丢失A组95、C组280，净增52；共同命中位置的N10贡献略负。它支持继续把问题限定为覆盖取舍，尚未证明query残差是损失根因。

现有四次从头训练都保持同一显式residual同时进入history和catalog；尚未检验“只保留catalog residual”的结构选择。下一臂改变这一条输入路径，保留已经产生部分收益的联合目录及原三loss，以回答：history侧显式residual在当前50k整体模型中是否值得保留？无论结果正、负或不确定，都能决定这个固定结构是否保留；它不是为了穷尽所有可能原因。

该依据允许提出一次有边界的比较，没有提供收益保证。移除history路径也可能损失用户协同表征；训练后的query、projection和catalog residual都会改变，不能把差异单独解释为冻结query的因果贡献。

## 唯一固定干预

由v5.2派生，令内容投影为p_i，seen mask后的catalog residual为r_i。当前v5.2在history token内容上加r_h；拟议v5.4仅取消该加法：

```text
v5.2 history input = SID embedding + projected_content + history_residual
                    + item/semantic position embedding
v5.4 history input = SID embedding + projected_content
                    + item/semantic position embedding

q = shared_encoder(history input) 的最后有效token表征
joint_logits_i = normalize(q) dot normalize(p_i + r_i) / temperature
loss = SID_CE + full_catalog_joint_content_CE + legal_prefix_mixture_NLL
deployment = 原joint_logits，按固定history资格排除已消费商品后稳定Top10
```

LayerNorm、dropout、history内容复制到四个SID token后的projection调用、单query、原联合catalog、learned alpha、cold residual0和原三loss各1保持。没有辅助native CE、独立第二encoder/query、teacher、外部推荐CP、pool、阶段续训、额外参数或新增loss权重。residual仍从零开始，在全部50k更新中通过catalog content CE及mixture内容路径学习；其SID loss经history的直接梯度路径随该干预取消。

“无history residual”不等于query纯语义：SID embedding、encoder与原多目标监督仍会学习交互信息。它也不等于恢复已有native参数或native排序，不能预先声称保住824个用户。

保持原训练配方：DDP2每卡128／global256，FP32，AdamW WD0.035，主干peak0.0003／residual0.002，warm2500／cosine50000／min0，clip1；相同SID、内容输入、12101目录、最多20历史商品、实际4层SID。seed固定42，自身100个raw dense Val点的首个N10最大值选own-best。完整部署Val仍单物理GPU→local[0]，不扫checkpoint或改变raw选择口径。等更新预算不代表等FLOPs、GPU时间或搜索成本。

## 相关工作与解释边界

[LIGER原论文§3](https://arxiv.org/html/2411.18814v2#S3)以历史encoder表征进行内容CE，并以条件decoder训练SID CE；两者共享历史表征。原Algorithm1使用生成候选并入cold后内容终排，本地mixture NLL属于额外实现，不能把它写成官方LIGER的learned-alpha融合。LIGER Appendix D的detach消融说明共享监督贡献随数据集变化，不能据此断言任何梯度普遍有害。

本方案定位为显式协同残差放置范围的固定结构比较；不把普通残差、共享encoder或消融本身宣称首创。共享目标的方向／尺度取舍是可能解释：[Yu等的原始论文](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)定义梯度冲突及额外条件，但当前没有实际梯度冲突证据，本方案也不采用gradient surgery。因果机制与新颖性均不能由最终Top10互补单独证明。

## 最小实施与必要验证（均尚未执行）

- `Liger.encode`与`dense_logits`当前共用`item_content_residual`。不能用residual_scale=0实现本方案，该开关会同时取消catalog residual及其optimizer参数。最小实现保持旧父模块不变：薄派生类覆盖该hook返回None以跳过history加法，并保留原小段`dense_logits`、在其中显式调用`CollaborativeResidualCoPMRec.item_content_residual(self, rows)`获取目录残差。只覆盖hook会同时关闭目录，必须验证这两条路径。不得用forward内临时修改模型属性的办法切换路径。
- 新版本的residual placement／checkpoint契约明确history=false、catalog=true，严格拒绝v5.2/v5.3错配；原shared-residual及dense CE支持集契约应在对应层一致表达。不能保留“history_and_catalog”旧字段却执行catalog-only。
- 新薄model／train-inference配置和根脚本仍经统一入口；writer metadata、checkpoint identity、配置记录和运行source实际归档保持一致，不新增旁路runner。历史408文件snapshot保留；实施时逐文件核旧字节和新增source，实际归档计数／哈希，不在proposal预填不存在的运行来源。
- 聚焦验证仅证明实现：零残差起点与原v5.2数值相同；非零残差时history输入不含该项、catalog分数仍包含；catalog residual有有效梯度且cold梯度为0；旧版本history默认委托行为保持；严格CP/配置／参数透传、Hydra compose和shell语法。实际CPU空optimizer、双rank一步smoke和正式source核验按新授权阶段协议执行，不能当作推荐收益。

## 最小正式对照、预测与结果对应决定

只有新候选42这一臂从头训练1次50k，再做1次own-best单卡完整Validation。复用已冻结native42及v5.2完整输出作匹配对照，不重跑baseline，不新增seed43、Testing、loss权重／LR／scale／temperature／alpha扫描。实际175文件／22363用户、labels／keys／history／目录、真实CP和source、合法唯一Top10先核验，再做原PCG64 seed42／2000次用户配对bootstrap。

主门禁不变：native42为分母，R10≥0.10470151589679383、N10≥0.0583728104417629，且两paired绝对CI下界为正。v5.2增量、native新增／丢失及共享命中位置变化作描述性对照，不添加parent硬门槛、不以事后分组替代整体门禁。

可检验预测是catalog-only输入有机会减少native命中丢失并提高净覆盖，同时保留联合目录N10优势；反向风险是丢失history协同表征，新增覆盖下降或推荐指标变差。不给收益或可恢复用户数预估，不另计算Top11或反事实分数。

- 主门禁通过：冻结该seed42完整方案，只称seed42合格；另行明确配对复现与Testing尚缺成本，整体目标仍未完成。
- 未过门槛但有清楚的部分增益：按用户既有要求保留对应证据，披露其他指标取舍；对v5.2的微小或CI跨0增量只称不确定，不自动晋升主方案。
- 对冻结v5.2没有明确推荐价值或出现明确负向：停止这个固定catalog-only结构，保留v5.2主方案及既有partial证据；不恢复辅助CE、不改成history-only、连续scale扫描或第二query，不追加下一臂预算。
- 实现／来源／输出有效性失败：仅修复可确认错误，保持唯一handle和实际成本，不把无效结果当成机制反证，也不自动重启完整训练。

当前只有这个可审阅proposal，代码、额度与正式结果均未产生。阶段累计成本和双8／复现目标保持原样。
