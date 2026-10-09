## Why

v4.3的有效历史资格层已在完整Validation确认自身R10+8.568824%、N10+17.367342%，188新命中、0丢失；保留这一层后，训练dense content CE仍以当前历史商品为竞争项，仅对cold设置-100。这提供一个具体的训练/推理支持集对齐假设，但尚未证明原CE支持有害或本改动有收益；fresh bad case的频率/content关系只作描述。

## What Changes

- 新增匹配的有界weights-only续训：两臂均从同一l3zyr91b v4.1 best6000出发，固定bias0、原三loss及learned alpha；treated只在训练dense CE原cold=-100之后把有效完整SID历史行设为负无穷，control保留原dense CE。SID CE与mixture NLL继续使用原定义/完整支持，不能称整mixture对齐。
- 两臂均按相同有效历史排除的dense Validation N10选checkpoint，保持主干LR.0001/item.002、WD.035、FP32、seed42、双卡globalbatch256、warmup300/cosine horizon6000，每臂max2000、每1000步验证；训练target不在当前有效causal history须实际验证。
- 新pool仅替换原control为各自Validation-selected新checkpoint：固定nj9elah1 source与新control各自由自己的history query产生全目录logits，0.5/0.5不变，继承原exact9历史推理资格规则。两新pool各做一次完整单卡Validation，复用已经审计的same-policy LIGER wdms8w77，不新增baseline Validation。
- root明确新增2训练×2000：旧5训练/30000封存，新累计限额7训练/34000，不重置；新完整Validation最多2次。按对自身旧history-pool≥3%且另一项无点退化判断组件保留，CI/固定分组限制主张；treated-control隔离history CE增量，control续训也可独立保留。
- 部署预先固定：满足对fair LIGER双10%、双paired CI下界正、旧绝对阈值及实际raw审计的臂中，选N10较高者；N10精确相同按R10、再按treated。只将唯一选定臂与同policy LIGER各做一次条件Testing，全线程Testing仍最多3（已用1）；两臂都不合格则不Testing、不扫描或续训，结束本阶段并保留已确认有效原层。

## Capabilities

### New Capabilities

- `copmrec-eligible-content-continuation`：仅训练content CE的有效历史支持排除、严格weights-only续训来源、匹配对照、固定history-pool评价和有界部署选择。

### Modified Capabilities

无。既有joint mixture、v4.1、fixed pool与history eligibility的默认行为和历史结题不改。

## Impact

新增薄recommendation续训/推理装配、model/experiment配置、根双卡训练与单卡推理脚本及聚焦测试；共享原helper、public checkpoint resolver/lineage、source snapshot与commonwriter，统一src.main/Hydra，无新依赖。当前仅规格与研究报告，不修改runtime、research-state/current-plan或启动任何实验；root review及规格完成之后才实施。
