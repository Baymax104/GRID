# 随机初始化基准证据盘点

## 结论

Beauty已有合格的单seed随机初始化对照mask_ce（y1dnupbt），可以与A（5g3wpbg7）隔离初始化差异；不需要重做seed42基准训练。当前明确缺口是该匹配基准的testing，以及seed43–46的匹配随机初始化训练。旧original TIGER与更早的TIGER不能直接替代这个对照。

## 源码与在线条件核查

重新读取W&B中catalog_arm为original/mask_ce的记录，以及旧Beauty TIGER 26qh50do、ye9u9yj7和A 5g3wpbg7的config、summary、验证曲线、输入产物和checkpoint。已记录的mask_ce训练仅Beauty seed42 y1dnupbt与Sports seed42 lhevwroq；未发现seed43–46的mask_ce记录。

当前代码中mask_ce和token_content_init均通过相同_teacher、conditional_log_probs、training_step和_beam，合法候选内归一化、损失reduction、搜索约束一致；均无content_query、辅助loss或内容混合参数。区别是A调用_initialize_tokens，mask_ce保留随机SID表。original则委托Tiger原始训练/forward/generate路径，不能将它和A的全部差距解释为初始化收益。

mask_ce与A的resolved config除arm、元信息/输出路径外一致：Beauty、RKMeans SID、同内容输入、seed42、devices=[0,1]、20k steps、每500steps验证、每卡batch128、Adam lr0.0005、相同模型/data配置、best val/ndcg@10选择、不训练后testing。两项各40个有效验证点，最佳step均19000。输入SID digest20f08b323a286fbb3f16b5ea27562af1、内容digestab56af975eac589c27eed6094482cbd7完全匹配。

mask_ce最佳checkpoint为checkpoint_epoch=000_step=019000.ckpt，Artifact digest14bee23f40882bef98b19685759afb35；A对应digest aaba5b47dc9a5e94c0d23b75cf697238。未重新审计历史远端源码与完整环境，因此是配置/来源/当前执行路径层面的匹配核查。

## 当前效果证据

| Beauty seed42 | 随机mask_ce | A |
| --- | ---: | ---: |
| 训练best-val NDCG@10 | 0.03999876 | 0.04256973 |
| 冻结evaluation NDCG@10 | 0.04000539 | 0.04257613 |
| 冻结evaluation Recall@10 | 0.07503466 | 0.08004293 |
| evaluation命中用户数 | 1678 | 1790 |

evaluation相对NDCG增益6.426%、Recall增益6.675%，净多112个命中。冻结evaluation数字与训练聚合略有差异，不能混算口径。

本轮复用既有已审计diagnosis文件q0ib4bh7（mask_ce）和n5rv1dgm（A），重新按user_id对齐22363用户，目标item逐位一致。目标前两层SID cluster bootstrap，2000次seed42：NDCG差0.00257074，名义95%区间[0.00026681,0.00474141]；Recall差0.00500827，区间[0.00143820,0.00860251]。这些是开发阶段evaluation上的单seed用户抽样区间，未做历史多重比较校正，不能替代训练seed不确定性或独立testing。此处数值来自本地既有证据重算，不冒充本轮新testing结果。

## 不能直接替代的旧基准

- j351huzp：同数据/20k/500step的original arm，best-val 0.03402611，但损失/搜索实现路径不同；只能作为系统级参照，不是纯初始化消融。
- 26qh50do：更早的Beauty RKMeans TIGER，验证间隔100而非500，训练后执行testing，组件/数据装配与A不同；需保留这些差异，不能当匹配基准。
- ye9u9yj7：更早的Beauty RVQ TIGER，量化器/SID不同、验证间隔100，亦执行过testing，更不能用于隔离A的初始化收益。

原始TIGER→mask_ce→A可展示不同改变对应的实验结果，但训练非线性且checkpoint分别选择，不能把差值当作可普遍相加的独立因果贡献。

## 最小后续方案（尚未实施或启动）

优先只补一次mask_ce seed42的冻结testing推理，使用现有19000步checkpoint、原mask_ce arm、与刚完成A testing相同协议，复用A seed42 testing sb6eapjd。无需新训练。该比较为完成A/残差testing之后补充的探索性对照，必须披露顺序，不能称预先注册确认，也不能将它与旧evaluation混合。

如果这一单seedtesting比较仍有明显同向收益，再考虑补齐mask_ce seed43–46四次训练，而不是重新训练A或加入模块。届时五seed均为同seed配对，保持原20k配置并按validation选checkpoint，统一报告所有seed。不能用一个seed42随机基准与A五seed均值相减声称跨seed收益。

如果单seedtesting未保留收益，停止扩展，报告泛化未确认；这不证明所有内容初始化无效。若保留收益，也仅有单seed支持，不能据此立刻主张完整论文已经成立。

论文现阶段可保留“完整内容均值初始化有单seed开发证据”这一受限判断，不能恢复“稳定优于残差”的主张。更宽的内容初始化主题也已有相关工作，补齐对照解决有效性问题，不自动解决新颖性问题。

原始快照与完整config差异：tmp/random_initialization_audit/runs.json、analysis.json。本轮未修改生产实现、未启动实验、未同步或提交代码。
