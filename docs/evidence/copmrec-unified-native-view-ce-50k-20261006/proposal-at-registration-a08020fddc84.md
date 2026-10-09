# 下一固定验证方案：共享query的无目录残差视图CE

## 状态与依据

本文件仅为可审阅方案：没有实施v5.3代码，没有启动新模型或评价，新增预算尚未分配。原阶段3次训练／150000更新、3次完整Validation已按真实成本封口；[五项路线判断与结题](copmrec-v5-2-stage-decision-20261006.md)先于本方案，不重置此前成本。整体目标仍是随机初始化、全部模块共同连续50k、单模型部署，相对同预算LIGER两个指标至少8%并完成配对复现。

保留v5.2：其相对native42 R10+4.658672%、N10+12.968595%，两个paired CI下界均正。剩余824个native命中丢失中，仅1个candidate列表包含cold；当前cold占位只涉及42个用户，单纯在固定分数下过滤这些商品最多影响42份列表，不足补足当前至少73个净命中的缺口。对native仍新增925、丢失824，说明两者存在明显seen命中互补，值得对seen商品的排序与覆盖取舍做一次固定验证。

这些事实没有证明query或残差目录是丢失根因。实际代码仍可检验一个具体区别：联合目录的CE可以通过残差目录拟合目标，却没有对同一query面对无目录残差视图的目标竞争单独施加CE。保留这个内容视图的直接监督是否有整体推荐价值，属于待检验假设。

## 唯一拟议干预

由v5.2派生，推荐模型从随机初始化开始；三个原loss、联合目录的训练／部署分数、完整目录支持集、learned alpha、共享query、seen历史与目录残差、cold residual0均保持。新增一个固定权重1的无目录残差视图content CE：

```text
joint_logits = normalize(query) @ normalize(projected_content + residual).T / temperature
native_catalog_logits = normalize(query) @ normalize(projected_content).T / temperature
loss = SID_CE + joint_content_CE + legal_prefix_mixture_NLL
       + cross_entropy(native_catalog_logits, seen_target_row)
```

辅助CE采用全部12101个目录商品作为分母，训练目标仍须seen，不增加history CE mask。训练decoder后只做一次商品目录projection，复用其实际dropout结果计算两个matmul，保留原随机数调用顺序。所有参数从step0共同训练，没有teacher、外部推荐checkpoint、阶段续训或pool。

这与v5.1的固定0.5／0.5 logits平均不同：原联合logits继续供CE、mixture和部署使用，辅助视图只用于新增CE，不改变最终scorer定义。[实际v5.1](../src/recommendation/liger/unified_dualview.py)、[实际v5.2](../src/recommendation/liger/unified_full_catalog_ce.py)。

“无目录残差”仅描述商品目录：history仍包含残差，query仍由共同编码器产生。辅助CE会更新共享query和projection，并经history路径更新残差；它不是独立原生LIGER模型，也不保证保存全部824个native命中。残差为零时两路logits相同，初始content监督因此加倍；单次实验不能区分语义视图监督与更大content CE权重的贡献，不将其包装成已隔离的机制。

不增加参数或部署计算，但训练增加一次完整目录matmul、CE及反向。每卡128用户×12101商品×128维时，该额外matmul的forward约0.397 GFLOPs；更新数相同不代表训练FLOPs、GPU时长或搜索成本相同。

相关工作定位沿用[LIGER](https://arxiv.org/html/2411.18814v2)的生成／内容CE联合建模，不将普通辅助CE宣称为新原理。这里拟检验的是共享协同残差后保留另一个语义目录竞争视图的整体价值，机制归因受本次单臂设计限制。

## 固定成本、有效性与决定

拟新增且仅新增：seed42的1次随机初始化连续50000更新训练，自己的raw dense Val N10选best，再做1次完整单卡Validation；新增Test0，不扫权重、温度、alpha、checkpoint或seed。该方案若分配，累计开发成本将为4次训练／200000更新、4个完整模型Validation，原启动失败仍单独保留。既有native42和v5.2完整输出作为冻结对照，无需重跑。

保持v5.2训练配置：DDP2每卡128／global256、FP32、AdamW WD0.035，主干peak0.0003／residual0.002，warm2500／cosine horizon50000／min0，clip1；使用相同SID、内容Embedding、输入及history资格，实际4层SID不变。推理仍一个物理GPU映射local[0]，单checkpoint。

实施需记录新增辅助loss及固定权重、训练支持集和版本，checkpoint严格拒绝契约错配；测试监督的实际梯度、单次projection／dropout顺序、零残差下两个logits相等以及固定参数下部署与v5.2一致。配置compose、根脚本参数、CPU空optimizer起点、双卡一步smoke、实际runtime归档及统一入口契约全部通过后才能正式启动。

主要有效性与门禁继续固定：22363用户和175原始文件、labels／keys／history／合法唯一输出／实际CP／source先审计；relative R10与N10对同seed native各至少8%，两个paired绝对差95% CI下界为正。PCG64 seed42、2000次bootstrap，不增加任意的v5.2非劣门槛代替主门禁；增量及NDCG取舍另报。

- 若主要双8门禁通过，冻结这一个seed42模型，明确seed43配对和Testing仍缺，随后再明确复现所需成本，不能宣布整个目标已完成。
- 若未通过但有清楚的部分推荐增益，依用户既有要求保留对应证据；微小或CI跨0的增量只称不确定，不将新增CE归因为已证明机制。
- 若相对冻结v5.2没有明确推荐价值，停止该固定辅助CE干预，保留v5.2；不自动扫lambda、复制第二query、继续续训或追加新预算。

本方案需要新增明确额度，当前累计账本没有可用训练或Validation槽。方案形成不表示额度已获分配；已有用户长期目标及原阶段结果保持原样。
