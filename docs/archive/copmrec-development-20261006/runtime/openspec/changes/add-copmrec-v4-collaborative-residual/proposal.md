## Why

v0在Beauty/seed42上的1608个Testing Top10命中仅比LIGER dense的1597多11个。无损恢复自身dense的97个候选遗漏仍不足10%目标。v3/v3.1候选修订及固定alpha没有稳定最终收益。本轮用户明确授权长时自主实现、SSH训练和单卡推理，允许改变模块；需要增加商品协同判别信号，而不重复已有全目录CE。

## What Changes

- 从学习alpha的v0 best48000显式weights-only初始化v4，不恢复旧optimizer/scheduler步数。
- 给seen商品添加零初始化协同残差，共享到历史输入、目录logits、内容CE、混合NLL及最终评分；cold残差强制零。
- 保持v0的SID/内容/混合三个loss和learned alpha、Mass候选beam20+cold；默认v0行为完全保持。
- 配置residual_scale=0作为同checkpoint、同训练预算的续训对照，1为试验臂；默认单卡推理。
- 新阶段先安排两次串行双卡短续训，各6000步；第三个训练槽仅在验证集有正向依据时用于确认/针对性干预，阶段累计训练最多3次、18000步。失败不重置额度。

## Capabilities

### New Capabilities

- `copmrec-collaborative-residual`: 以v0为来源的冷商品安全协同残差及严格checkpoint契约。

### Modified Capabilities

无。

## Impact

修改Liger共享投影的最小hook，新增v4 model/experiment/script、CPU单测和结果记录。运行始终经src.main/Hydra，输入和产物遵守已有Artifact与bundle协议。第一轮评估仅使用evaluation验证集选checkpoint；Testing不参与调参，达到门槛必须复算完整输出并和042139al配对比较。
