## Why

当前 CoPMRec 两成员的共享生产链实际64k更新，旧LIGER只有50k；原日程末期平台不能排除非零学习率重启收益。用户明确要求开始补预算匹配对照，以判断同更新预算、同两成员机会下是否仍保留推荐收益。

## What Changes

- 新增 native LIGER 的A/B/C三段仅权重恢复训练适配：6k/6k/2k，各fresh AdamW与scheduler，原两项loss保留。
- 新增真实native LIGER A/C各0.5、独立query、同history资格的固定dense logit pool，及C单成员推理配置。
- 新增薄Hydra实验和根脚本，统一src.main、双卡训练/单卡推理、source snapshot、notes/dry-run/额外override契约。
- 新预算明确登记3训练/14k、2完整Validation、最多1新Testing；旧7/34k与Testing3/3不重置。完成实际checkpoint及独立输出审计后判断原10%主张。

## Capabilities

### New Capabilities

- `liger-budget-matched-continuation`: 三段native weights-only continuation、来源与实际预算验证、固定双native LIGER融合及公平比较。

### Modified Capabilities

无；现有LIGER、CoPMRec及其归档结果不改变。

## Impact

新增recommendation模型及最小CPU tests、model/experiment配置和根脚本。复用公共Artifact loader、history helper、writer、launcher与source snapshot；不加依赖，不新增独立训练/推理入口。node1使用明确物理GPU映射，运行前Mutagen官方flush及完整源字节核验。用户本次“开始”授权执行这个固定对照，不授权额外方法或参数扫描。
