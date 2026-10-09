## Context
骨干已经分别实现；训练协议须遵循Linear的共同split、history20、FP32、validation NDCG@10选checkpoint。
## Goals / Non-Goals
形成独立的Lightning模块及恢复校验；不运行正式实验，不自行训练来源不明的CF teacher。
## Decisions
tokenizer显式输入keyed32维CF，随机种子控制group采样；全目录初始化和碰撞评价。推荐模块输出原始item key；独立torchmetrics按用户累积，DDP只在compute归并。checkpoint检查目录、模型超参和输入指纹，拒绝替换同shape目录。
## Risks / Trade-offs
官方未发布可复现CF teacher训练实现，CF需提供训练split来源证明。碰撞修复失败阻止SID导出。推荐温度仅影响训练loss，生成使用原始logits。
