## Context
Linear BMX-58要求Beauty/Sports/Toys与三种子，共同split、history、全目录、val NDCG@10最佳checkpoint；BMX-6核心方法超参不能直接套到LETTER。
## Goals / Non-Goals
提供完整可运行LETTER链路并验证dry-run和5step；不开始正式9次训练，不宣称复现论文效果。
## Decisions
推荐采用共同history20、FP32、两卡、每卡128、50k步和500步验证；保留官方T5结构、AdamW lr5e-4 WD0.01、warmup1%和cosine、temperature1.0。tokenizer采用作者独立epoch协议。CF是必填显式输入并记录来源。正式推荐只消费新tokenizer输出的4码SID。
## Risks / Trade-offs
论文效果须等待真实CF来源和正式运行；有界GPU验证用标明的synthetic fixture不作为效果证据。远端通过受控Mutagen同步，禁止安装PyTorch/CUDA。
