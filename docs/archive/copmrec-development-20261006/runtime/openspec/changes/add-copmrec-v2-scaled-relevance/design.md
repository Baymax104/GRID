# 设计

## Context

沿用v0共享encoder、内容投影、SID decoder、合法前缀mass及可学习融合参数。v1保留历史结果、v1.1为其最后实现。v2直接继承JointMixtureLiger，不继承v1推荐组件，后续版本为v2.x。

## Decisions

- head为LayerNorm(d)→Linear(d,128)→GELU→Linear(128,1)，末层零初始化。d读取[start,完整SID]后的decoder最终末位hidden state。只输入d，不拼接h/v。
- c=cos(h,v)/temperature；sigma=sqrt(population_var(c over natural_candidates)+epsilon²)，epsilon=1e-3，统计采用float32并detach。0/1个自然候选的方差按0处理，避免NaN。
- 自然候选固定为当步生成beam与全体cold并集；训练扩展候选的content Top20与真实正例不改变统计集合。训练与推理共享同一尺度/评分函数，统计跨chunk计算，不按chunk归一化。
- r=beta*sigma*tanh(head(d))，beta固定0.5；score=c+r。有限非负beta及有限正epsilon可配置，新配置将默认值显式记录。beta不是优化参数，没有尺度惩罚或独立head CE。
- L=L_sid+L_content+L_mixture+lambda*L_fused_CE，lambda固定0.01，不做自适应/暖启动权重调度。每rank每微批抽4例，融合CE对自然候选+content Top20+正例去重列表计算。
- 跨用户合并(user,item)对，每chunk256次decoder评分，所有共享表示保留梯度；只有beam离散选择与尺度统计停止梯度。单一AdamW，无冻结teacher。
- 新版本保存严格的score/head、beta/lambda/epsilon/chunk、catalog和评价历史契约；拒绝v1/v1.1恢复。支持可选v0 weights-only后共同训练，默认两个checkpoint引用null。
- 独立v2 trainer以500微批验证、check_val_every_n_epoch=null、累积1及每50更新训练日志，对齐BMX-116真实v0配置。用户随后确认全evaluation/testing链路统一，保留融合hybrid选点、DDP安全设置与共享MetricCallback步数语义；v1闭合开发行为显式隔离，不改写历史记录。
- 日志包含raw/weighted ranking loss、自然content尺度、相关性绝对值、自然候选相关性/content尺度比例及tanh饱和率；不使用这些诊断替代Recall/NDCG。

## 风险与验证

幅度上界为beta*sigma，并非严格保留v0排名；小分差仍可能翻转。head受tanh饱和限制，候选覆盖缺失和任务梯度冲突仍可能存在。基础三项和最终融合CE训练同一主干，正例注入只保证loss可计算，不直接为beam离散覆盖提供梯度。

检查自然统计对标签/训练扩展独立、偏差修正0、空/单候选、分片不变、d-only与完整SID/user映射、上界和梯度、非零dropout连续更新、两进程同步、checkpoint/trace、标签独立以及配置脚本quoting/override/LF。

## 研究边界

本次实施授权来自用户，完整训练由用户手动开始；agent新增完整run与训练实验预算0，最小CPU单元更新不是效果实验，不重置已关闭的2000更新校准阶段。当前问题仍是生成表示能否转为最终推荐收益。用户手动运行后以真实v0为必要效果对照、同候选content-only为机制对照，全evaluation按各自分数选点后在同一testing独立评价；匹配收益改善才保留候选，结果负向则不晋级该实现，口径不匹配或尚未训练则未知。此处不自动追加对照或权重扫描。
