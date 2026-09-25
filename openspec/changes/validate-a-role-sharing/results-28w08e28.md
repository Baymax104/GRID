# A角色共享诊断结果：未通过推进门槛

日期：2026-09-18。正式run：[28w08e28](https://wandb.ai/baymaxam/GRID/runs/28w08e28)，状态finished，非dry-run。结论：本次局部诊断不足以支持角色共享构成值得立项的瓶颈，暂停结构推进，不进入多seed或短续训；不等于证明共享在所有条件下无害。

## 身份与证据质量

- Beauty / RKMeans / A full_content / seed42，协议role-sharing-v1。32名training支持用户生成方向，64名互斥evaluation用户检验，13个条件，无优化器更新。
- checkpoint来源5g3wpbg7，SID来源4vyi4o6w，embedding来源3jtt9mpa；W&B已记录三个上游Artifact的使用关系。
- 输出Artifact：`baymaxam/GRID/tiger_a_beauty_role_sharing_audit_seed42-evidence:v0`，文件`role_sharing_audit.pt`。
- 文件SHA256：`389e69ed7f1101768b9cad67d75ff522021eeeff8847b01ebde6be311dd86541`。
- checkpoint指纹：`15547637d02ca842fe00adaf2dde8701fc11ea0585f889035432f100319b2e03`；catalog指纹：`100f0b13ce2d08e8895f17db9fa1aff3951d03c8c7abdaf7faee4af268b7663e`。
- 公共证据validator通过；support/evaluation交集0，最大梯度分解误差3.73e-9，最大概率质量误差2.38e-7。无Top10容差边界歧义。
- 与G0 ln1f2l00交叉核对：checkpoint/catalog一致，64名评价用户全部重叠，输入hash与baseline精确排名一致。因此是重复使用的开发样本，不能写作独立确认实验。

## 配对效果

CE为每SID token平均负对数概率，baseline为2.114333792。下表改善量为reference CE减candidate CE，正数有利于candidate。按评价用户配对bootstrap，20,000次，seed20260918；95%为未做多重比较调整的边际percentile区间。

| 比较 | 相对扰动幅度 | CE改善 | 95%区间 |
|---|---:|---:|---:|
| split相对shared | 0.001 | 0.00012432 | [-0.00001792, 0.00027957] |
| split相对shared | 0.003 | 0.00032670 | [-0.00008534, 0.00076915] |
| antisymmetric相对random_antisymmetric | 0.001 | 0.00017060 | [-0.00010441, 0.00045384] |
| antisymmetric相对random_antisymmetric | 0.003 | 0.00050031 | [-0.00032637, 0.00134930] |
| split相对prefix | 0.001 | -0.00000717 | [-0.00043926, 0.00039418] |
| split相对prefix | 0.003 | -0.00003481 | [-0.00125746, 0.00110683] |

两个幅度的split相对shared点估计方向一致，但区间均跨零；反对称方向也未可靠胜过随机对照。split甚至未显示超过prefix-only的独特优势，不能据此证明所有收益来自prefix。

13个条件均为2/64命中，Hit@10=0.03125，NDCG@10=0.0176707774；目标进入/退出Top10及Top10内目标名次均未改变。split在两个幅度下分别改变20/35名用户的Top10列表，说明干预并非无效执行，但没有转化为观察到的目标推荐收益。split相对baseline的全目录目标排名改善/变差人数分别20/28和22/34，也没有一致排名改善。

## 梯度与解释边界

前三层聚合历史/前缀梯度余弦分别-0.00636、+0.00531、-0.00287，接近正交，未显示强且一致的对抗。支持集中，同用户同token两条路径都活跃的组合仅8个，其中1个负内积；不足以刻画系统性冲突，也不能排除跨用户干扰。

第四层前缀梯度为零是结构性质：预测最后一层无需把该层目标token作为有效前缀输入。第一层history曝光含encoder已有的masked-target占位，不能解释成纯历史item频次。

本实验只观察一个checkpoint、两个小扰动幅度及64名开发用户；只有2个Top10命中，排名指标敏感度有限。样本bootstrap的零排名差不代表总体真实效应必为零。局部敏感性也不能替代从头训练或短续训结果。

## 决策与下一步

技术有效性通过，预设效果门槛未通过。当前证据不足以把“有害角色共享”写为论文瓶颈，也不足以为拆表、adapter、gate或梯度处理立项。

建议到此暂停该候选，保留实现和证据，不自动扩样本、扫幅度、补seed或短续训。A保留强基线，互补保持停止，G1/CCFD保持暂停。下一轮先从现有失败案例寻找有可观测证据、能区分竞争解释的问题；没有新证据前不直接实现下一个模块。

原始下载及分析脚本位于`tmp/role_sharing_28w08e28/`；完整派生统计归档于[研究证据JSON](../../../../research/docs/evidence/2026-09-18-role-sharing-28w08e28.json)。本轮仅读取W&B和离线分析，不启动模型实验。
