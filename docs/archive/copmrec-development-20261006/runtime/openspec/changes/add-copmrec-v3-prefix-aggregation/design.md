## Context

v0 使用 `JointMixtureLiger`，所有层通过后代 logsumexp 计算内容条件概率，训练使用 teacher forcing 混合 NLL，推理使用一次 beam20 搜索和 content 终排。用户指定 v3 从真实 v0 派生，训练配置与 v0 一致；完整训练手动开始。

## Goals / Non-Goals

**Goals:** 首层 Max、后续 Mass 同时用于训练混合项与推理；独立版本和恢复契约；保持 v0 的随机初始化、全部推荐参数可训练、优化器、数据、dense验证选点及双卡batch。

**Non-Goals:** 不加入 head、排序loss、teacher、历史排除或候选合并；不调整 alpha 参数化、beam预算或验证评分；不启动完整训练，不把实现通过称为收益。

## Decisions

1. processor 新增固定 `root_max_mass`：第0个SID决策的后代聚合取max，depth>=1取logsumexp；在合法兄弟节点内log_softmax。后续层分母来自本层兄弟mass总和，不能用根层max或上一层混合聚合值归一化。
2. `JointMixtureLiger` 仅新增训练processor工厂扩展点，默认仍mass；v3子类覆盖工厂与推理processor，共用原三项等权损失和全局alpha。保持 state_dict键、参数数量及初始化随机序列一致。
3. v3配置继承v0 train/inference和model，只覆盖模型target、固定聚合及运行身份。trainer仍原 `ddp`，验证仍dense；不从v2继承hybrid选点、DDP调整或排序目标。
4. checkpoint独立保存v3版本、逐层聚合、单位权重目标契约；缺失或不匹配时拒绝恢复。trace记录聚合列表及checkpoint alpha；保持现有bundle/writer接口。
5. 脚本调用原 `liger_launch`，支持notes、dry-run和末尾override。从零双卡命令明确 `ckpt_path=null`、物理GPU2/3映射逻辑[0,1]，每卡128、累积1、有效batch256。

## Risks / Trade-offs

- 根层变化会改变后续frontier与累计分数，不能把Max和Mass的局部优势相加；后续评价完整新增、损失及排名变化。
- max只向最佳后代传梯度是本版定义；验证梯度有限和共享推荐参数得到更新，不更改基础content CE。
- 同配置不证明效果或GPU吞吐；分别报告CPU数学/恢复、Bash/Hydra、Linux双进程验证。
- testing已经用于开发分析，旧v1/v2关闭预算保持；本次只准备用户手动运行的50k配置。

## Migration Plan

新增v3入口并保持原版本入口；通过Mutagen受管范围同步，flush后核验三会话及运行文件hash。回退可选原v0入口，不改写旧checkpoint/Artifact或研究证据。

## Open Questions

最终候选净收益未知；实际训练后按dense验证选best，在相同testing上比较v0/LIGER dense并报告候选路径损益。
