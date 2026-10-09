## Context

backbone 和 data 已独立完成。MetricCallback 管理 train/val/test，但不处理 predict；统一推理入口使用 Trainer.predict。需要薄 prediction callback 复用同一 MetricEngine 与 logging_modes，不能在模型内直接写 W&B。

## Goals / Non-Goals

**Goals:** 官方训练梯度、全目录准确排名、用户/商品 key 输出、可审计恢复。

**Non-Goals:** 不新增损失变体、历史过滤、sampled evaluator 或真实训练预算；pipeline 配置留到下一模块。

## Decisions

- training_step 返回 loss payload，使用原生逐位置正负 BCE。optimizer 必须来自 TrainingModelConfig；后续配置使用官方 Adam(lr=0.001,betas=[0.9,0.98],eps=1e-8)，不默认 scheduler/weight decay。
- validation/test 不生成随机负样本，也不计算会改变 RNG 的 sampled loss；只对末尾目标计算排名。prediction 同一算法，全目录保留冷启动商品，不过滤历史，匹配现有 LIGER 评价语义。
- 分块 scores，每块与当前 TopK 合并；stable 降序排序保证同分按固定目录升序，结果与直接全库排序一致。只保留 O(B*(chunk+K)) scores；逐商品点积沿 hidden 维固定归约，临时乘积为 O(B*chunk*D)，避免 GEMM 块形状使数学同分出现微小偏差。
- metric adapter 将原始商品 key 相等性转为现有 Recall/NDCG 所需 preds/target/indexes。predict callback 复用配置同一 test MetricEngine，logging_modes 仍由 MetricCallback 参数控制。user_count 为分布式可求和 metric。
- checkpoint 保存 catalog SHA、原始 keys、官方 commit 和结构协议；load 拒绝目录或历史/算法配置变化。仅 chunk_size/top_k 可变化，不改变已训练权重。

## Risks / Trade-offs

- 原始模型随机 embedding 对冷启动没有内容泛化 → 完整保留评价，不缩小目录或删除用户。
- 同分排序与 torch.topk 不稳定 → 固定 key 升序作为 tie-break，训练不受影响。
- prediction logging 不支持 Lightning log → history 通过 logger.log_metrics；summary 沿用 MetricCallback 机制。
