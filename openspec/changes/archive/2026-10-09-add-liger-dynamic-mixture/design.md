## Context

固定混合有正向候选证据，旧残差排序实例结题。用户明确切换到动态混合；sampling 暂存。复用冻结26k LIGER，采用原 hybrid 候选链路。

## Goals / Non-Goals

目标：检验用户/前缀状态门控是否超过固定0.5及可学习常数。非目标：搜索beam预算、sampling、dense20并集、独立排序网络、backbone训练。

## Decisions

- 共享特征函数：归一化层级、两路合法分布归一化熵、两路top1-top2差、JS/log2、top1一致性，共7维。线性sigmoid8参数；constant只学bias。零初始化。分支权重共享，死beam全负无穷，单分支不改变概率。
- 同一ProbabilityMixtureProcessor提供训练和推理条件概率，训练后目标token才被gather；防止目标进入特征。全词表teacher logits在合法子分支重新归一化。
- 只缓存training最后目标四个前缀的7维特征和两路目标log概率，单机本地bundle及manifest、SHA与源checkpoint/目录身份；无W&B缓存发布。
- 用户sha256(42:user_id) mod10留出10%，全部训练目标参与，不按候选覆盖筛选。两臂各3epoch、batch256、AdamW lr.01/wd0、seed42，最小fit1000/val100；按内部val/nll最小选checkpoint。
- 最多5次完整run：1缓存+2门控训练+2实际beam评价。复用已验证固定0.5结果，同源核验后比较；若内部val动态NLL不低于constant及固定0.5，停止在3run。通过才运行两次evaluation；不使用testing调参。
- 最终动态需同时优于固定及constant的NDCG10配对95%区间下界>0且Recall点不降；仅NLL改善不晋级。全库dense只作参考。缓存和两次beam成本需实测；不训练backbone，缓存约N*4*9*4字节加keys，门控每臂ceil(fit/256)*3步。旧86秒混合预测仅量级参考，非本次耗时承诺。

## Risks / Trade-offs

真实前缀与beam前缀分布不同：必须分阶段检验实际候选与最终指标。全库内容打分仍在：不宣称扩展性收益。门控退化为常数：保留constant对照。最后一层经常单分支：真实分布特征和损失仍保持有限。缓存不含evaluation；来源身份不匹配则失败。

## Migration Plan

新增配置隔离。固定接口无门控时保持原结果；新gate在基础checkpoint恢复后安装，避免污染旧state_dict。仅通过统一入口和根脚本运行，完整实验由用户手动启动。


## 用户授权的一次预算扩展（2026-09-24）

前一阶段5/5已结束且动态增量未建立。用户明确授权一次更长训练，若常数仍更好则采用常数。新增最多4run（两臂训练+两臂beam评价），同一问题累计最多9run；不重建缓存、不训练backbone。

两臂均从零初始化重跑，保持seed42、batch256、lr.01/wd0、特征和损失；max_epochs30（每臂最多2370步），每epoch内部val NLL，EarlyStopping min_delta=.0001/patience5/mode=min/check_finite=true。按所有epoch原始val NLL最小值保存最佳checkpoint，不按evaluation选epoch。重新初始化避免恢复旧callback路径与数据顺序差异，前3epoch应与旧曲线数值接近，审计时核对。

无论动态内部NLL是否领先，按本次授权完成两臂最终比较；这次专门检验预算不足解释。主要比较动态对constant的NDCG10，配对95%下界>0且Recall10点不降才保留动态；常数更好或无法证明动态额外收益则采用可学习常数，统计不显著不称等价。不再因未收敛扩第二轮预算。所有结果与旧三epoch结果均保留，testing不使用；sampling仍暂存。
