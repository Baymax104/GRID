## Context

原 C 为 `content_init_full`，Beauty seed42 最佳 validation NDCG@10 为 0.0413907468（`k6jvoo2v`），A 为 0.0425697267（`5g3wpbg7`）。完整分支具有内容初始化和辅助 CE，不能由这一差距直接推断内容路线无效或存在梯度冲突。研究依据见相邻 change 的 `refinement-review.md`。

## Goals / Non-Goals

目标：在同一 CGBS 架构内形成一个可归因的优化路径对照，保留 query 的辅助监督。

范围之外：调损失权重、改温度、增加新模块、冻结整个编码器、自动启动完整实验，以及声称已证明梯度冲突。

## Decisions

1. 新增 arm `content_init_full_aux_detached`，由已有 arm 字段记录到 resolved config 和 checkpoint contract，不增加影响历史 C checkpoint 的契约字段。
2. 生成路径使用 `q_mix = query(encoded)`；辅助路径使用同一 MLP 的 `q_aux = query(encoded.detach())`。不能对 `q_mix` 本身 detach，否则辅助监督无法训练 MLP。原 C 仍复用一次 query。
3. query 当前只有 Linear/GELU/Linear/normalize，无 dropout、running statistics；再次计算不会推进随机流。固定权重下前向值及 query 梯度保持一致，训练后的参数轨迹预期不同。
4. 使用现有脚本 `--condition d` 启动一次从头训练，GPU 0、1；默认 `both` 保持 B/C。D 推理使用原有 trained/off/exact/shuffled_exact/rerank，不修改打分公式。
5. 用 CPU 小模型验证辅助损失在编码器输出与参数上的梯度被移除，query 梯度保留，生成梯度相同。运行时不添加额外 decoder forward 或 DDP autograd 探针，避免额外随机消耗、计算开销和同步风险。

## Risks / Trade-offs

- 隔离可能切断有益梯度 → 单次 D 与已有 A/C 比较；结果不达标也不能直接证明不存在梯度冲突。
- 前向等价容易被误解为训练等价 → 仅声明同权重、同输入、同随机状态下等价，D 的优化轨迹本来就会变化。
- query 将来加入随机层会破坏等价 → 测试覆盖随机数状态和带 backbone dropout 的前向比较。
- seed42 不能支撑稳健性结论 → D 超过 C 只作为改进信号；还需固定 checkpoint 的 trained/off 对照及后续独立 seed 确认。

## Migration Plan

新增 arm，无历史数据迁移。聚焦验证后按项目 Mutagen 流程同步；原 C 仍可直接启动作为回退。用户手动启动 D。恢复与推理必须匹配 D checkpoint 身份。

## Open Questions

实际梯度冲突是否存在、D 是否超过 A、在线内容分支是否在 D 上产生增益，均待实验；本改动不采集运行时梯度夹角或独立纯 token loss。
