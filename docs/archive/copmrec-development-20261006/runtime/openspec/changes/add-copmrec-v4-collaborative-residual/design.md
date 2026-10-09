## Context

当前v0已有全目录内容CE，新增普通InfoNCE会重复目标；表示由共享内容MLP和SID token构成，没有商品ID协同残差。候选入口修复没有提供足够10%收益空间，固定0.8训练损失49个命中，其中dense损失46个。新假设是增加训练商品的可学习协同自由度可以改进目标与相似内容竞争商品的区分。

## Goals / Non-Goals

目标是从v0派生、保持初始化推荐等价，验证残差能否提升最终hybrid推荐。要求Recall@10与NDCG@10相对真实LIGER dense均至少10%；未达门槛仍保持全线程目标active。实验严格分开验证选择与Testing确认。

不修改v0默认、不冻结alpha、不重新量化SID、不增加离线训练入口、不使用Testing标签训练，不把执行通过称为推荐收益。

## Decisions

1. 共享投影后加r_i，训练cold mask强制r_i=0；同一残差供历史输入和目录打分，保留cosine/temperature。
2. 用torch.zeros构造参数，不消耗RNG；r_i=0时状态、loss和推理与v0一致。residual_scale=0冻结残差参数。
3. 专用v4 subclass校验v0来源version/learned policy/catalog SHA/state keys，再完整严格加载旧权重与新零残差；记录来源URI、文件SHA和旧global_step。
4. v4恢复只接受v4/version/scale/协同协议匹配的checkpoint；pretrained路径只用于训练初始化，推理/恢复使用ckpt_path并置初始化null。
5. 先用物理GPU2,4（logical0,1）训练，推理仅物理GPU2（logical0）。训练batch128/GPU、AdamW主干1e-4/残差1e-3、weight_decay0.035、warmup300、cosine6000、FP32、seed42；hybrid验证每1000步，以val/ndcg@10选best，完整evaluation用户、beam20。DDP关闭buffer broadcast，指标全局同步，不补齐重复验证用户。

## Risks / Trade-offs

增加约1.55M参数且多6000步训练，收益不能与匹配续训对照混淆。seen残差可能过拟合，cold不保证提升。保持候选beam20可能漏掉改善后的dense目标；先验证实际最终指标再决定是否需要调整候选机制。旧v0训练源码provenance缺口保持。

## Migration Plan

新增独立v4入口，所有旧版本默认不变。通过CPU等价/梯度/冷商品/严格加载测试、Hydra compose、shell参数和OpenSpec检查后Mutagen flush，核对remote source SHA，真实双卡dry-run后正式串行启动。W&B记录config/notes/checkpoint/source和实际进程句柄。

## Open Questions

最终10%效果未知；残差是否改善候选外增量未知。正向验证支持Testing确认，负向停止该具体模块，不自动扩大扫描；第3槽仅在现有正面依据有决策价值时使用。
