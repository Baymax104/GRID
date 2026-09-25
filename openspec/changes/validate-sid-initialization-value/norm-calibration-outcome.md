# 深层范数校准两seed结果

## 结论

固定eta=0.5候选未通过预先约定的诊断门槛：seed45/46最佳NDCG均下降，最终步、后五次均值及最佳checkpoint的Recall同向下降。停止该候选，不补其他seed、不自动扫eta、不增加在线模块。保留原A，不据此宣称所有范数设计都无效或原A范数已最优。

## 在线核验

本轮通过W&B Public API重新读取两候选及两个A参考run的config、summary、notes、40个有效验证点和Artifact记录。候选筛选条件为initialization_protocol=deep-norm-calibration-v1且run_mode=train，返回两项，均finished。

| seed | A run | 候选run | A/候选最佳step |
| --- | --- | --- | --- |
| 45 | [kckgwr6j](https://wandb.ai/baymaxam/GRID/runs/kckgwr6j) | [t7holxig](https://wandb.ai/baymaxam/GRID/runs/t7holxig) | 20000 / 20000 |
| 46 | [v7q7y0jx](https://wandb.ai/baymaxam/GRID/runs/v7q7y0jx) | [11ndc0b5](https://wandb.ai/baymaxam/GRID/runs/11ndc0b5) | 19000 / 19000 |

每项从头训练20000步，每500步验证，每卡batch128，lr=0.0005，逻辑devices=[0,1]，不训练后自动testing。候选model.root明确记录deep_norm_calibrated与initialization_norm_exponent=0.5，参考run与pair_seed匹配。

完整resolved config差异限于初始化方式、指数、元信息/输出路径，以及seed46的物理卡号。seed45两者物理GPU0/1；seed46历史A为2/3、候选为0/1，不能称完全相同硬件执行。两卡拓扑及逻辑设备配置一致。本轮未独立审计远端环境、源码hash或下载checkpoint反查内部权重。

四项输入均为SID rkmeans_inference-semantic-id:v2（digest 20f08b323a286fbb3f16b5ea27562af1）及内容sem_embeds_inference-semantic-embedding:v5（digest ab56af975eac589c27eed6094482cbd7）。本地norm-calibration-implementation.json快照全部hash仍匹配。

## 成对指标

差值方向均为候选相对A，百分比为相对变化，非百分点。

| seed | A最佳NDCG | 候选最佳NDCG | 相对变化 | A最佳点Recall | 候选最佳点Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 45 | 0.03845856 | 0.03627887 | -5.668% | 0.07341661 | 0.06930223 |
| 46 | 0.03836792 | 0.03757816 | -2.058% | 0.07341952 | 0.07131361 |

| seed | A最终NDCG | 候选最终NDCG | 相对变化 | A后五次均值 | 候选后五次均值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 0.03845856 | 0.03627887 | -5.668% | 0.03787409 | 0.03571168 | -5.709% |
| 46 | 0.03717608 | 0.03712192 | -0.146% | 0.03729586 | 0.03703416 | -0.702% |

seed45候选仅3/40验证点领先，最后10点0/10领先；seed46为15/40、最后10点3/10。前者为持续后段退化，后者大体接近但未出现稳定收益。后五次Recall均值也分别下降0.00404195、0.00082448。相关验证点不是独立统计样本，不将这些计数用于显著性检验。

## 产物身份

| run | 最佳checkpoint文件 | Artifact digest |
| --- | --- | --- |
| t7holxig | checkpoint_epoch=000_step=020000.ckpt | a3828aa5f0e211ee1a5a848f2cae8c54 |
| 11ndc0b5 | checkpoint_epoch=000_step=019000.ckpt | cecc6bc82653abc69755ba6d5c3744d5 |
| kckgwr6j | checkpoint_epoch=000_step=020000.ckpt | 0b0efe97e733de5ae998d3eb242677ae |
| v7q7y0jx | checkpoint_epoch=000_step=019000.ckpt | 658aff9de49911557ac5a6443847ab02 |

Artifact metadata的selection=best、monitor=val/ndcg@10、best_model_score与曲线最佳值一致。文件显式记录，可避免best/last歧义。此次只读核验，不发布、删除或启动run。

## 理论更新和下一步

静态分析建立了深层范数差异较大这一事实，训练结果不支持将它解释为可由平方根压缩修复的瓶颈。压缩范数可能削弱有用的一致性信息或扰动合适的优化先验；这仍是解释，不是因果机制证明。本轮也不能反向推出增加范数差异必然有益，不能据此直接提出eta>1或新正则。

原A保留为当前工程主候选，机制叙事收敛到完整内容条件均值作为SID初始化；未证实新的校准机制。若接受停止继续调A，下一步应锁定原A和匹配残差对照的五seed最佳checkpoint，先落实独立testing验证，不用testing继续选方法。本轮未打开testing或启动额外推理。两seed为事后选择的诊断案例，所有数值来自参与checkpoint选择的evaluation，不是独立测试结论，也不声称统计显著。

## 可复查文件

- tmp/norm_calibration_results/runs.json：本轮W&B只读快照。
- tmp/norm_calibration_results/analysis.json：成对指标、完整配置差异与产物身份。
- tmp/norm_calibration_results/curves.png：两seed验证曲线。
- tmp/inspect_norm_calibration.py及tmp/analyze_norm_calibration.py：轻量抓取和本地汇总，不经过Trainer、不产生实验run。
