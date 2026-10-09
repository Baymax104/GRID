## Context

原 v4 的 3×6000 steps 和 v4.1 的 2×6000 steps 均已关闭，累计 5 / 30000 的训练预算用尽。fresh source/control 配对有 320 新命中 / 258 丢失，固定 639 个 offender 真目标用户净损 48、其余 21724 用户净增 110，两个固定 key 组整体净增 43 / 19。这支持唯一一次固定推理验证，不能推出 pooling 实际有效；细节与五项门禁见研究报告。

## Goals / Non-Goals

**Goals:** 两个已固定成员各自产生完整目录 logits，固定等权融合、稳定排序、单卡标准输出；严格校验来源、catalog 与原模型恢复契约，可独立复算一次 Evaluation。

**Non-Goals:** 新训练、学习或扫描权重、temperature、normalization、pair、checkpoint、bias、LR；Top10 候选/分数拼接、概率算术平均、模型参数平均；Testing 选择规则或参数。

## Decisions

1. 新 `src/recommendation/liger/fixed_logit_pool.py::FixedLogitPoolCoPMRec(LightningModule)` 仅推理。constructor 采用 keyword 参数 `catalog`、`source_model_factory`、`control_model_factory`、`source_checkpoint`、`control_checkpoint` 及四个必填 identity：`expected_source_reference`、`expected_source_sha256`、`expected_control_reference`、`expected_control_sha256`。模块验证 identity 格式和 loader 注入的实际 reference / raw SHA 一致；生产 pair 固定在配置、ready / launch 和独立审计，不在模型中实现 W&B 解析。
2. 生产 source 为 `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，SHA `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055`；control 为 `wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，SHA `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9`。启动前配置比较拒绝替换两者或评分设置，不提供 pair / weight / temperature 搜索。
3. catalog 只通过 `load_liger_catalog` 公共入口读取一次。两个 `_partial_: true` factory 分别使用原 `CollaborativeResidualCoPMRec` 和 scale0 的 `ItemScoreBiasCoPMRec`，dense/content、residual scale1 / multiplier20、pretraining 为 null；显式向两者传同一 catalog。保持原成员的架构与评分设置。两份 checkpoint 均经 `load_copmrec_pretraining_checkpoint` 读取；分别调用各自 `on_load_checkpoint` 和 strict state load，保留原 v4 / v4.1 版本，不桥接、不恢复 optimizer / scheduler / Trainer step。
4. 验证两成员 item_keys、semantic_ids、content_bank、seen_mask 逐值一致并等于公共 catalog；来源结构、两个 best6000、learned alpha / joint contract、residual multiplier20、cold residual0、control bias 全零及所有 state 有限。两者 v0 来源元信息完全相等且为 v0 best48000 weights-only；control 的 v4 warmstart 必须精确指向 source 的 reference / SHA / step6000。生产原 v0 URI / SHA `567aed4610a2cfe6671a1eddd300470184d402853269559a3e719da95ba9a322` 同步由 ready / audit 核验。
5. 每个成员对同一历史独立 `encode → query → dense_logits`，覆盖同一完整目录；固定算式 `0.5 * source_logits + 0.5 * control_logits`，不额外归一化或使用 labels。按降序、catalog row 稳定处理 ties 后取 Top10 完整 SID。所有参数冻结、成员 eval；fit、training_step、configure_optimizers、`train(True)` 与独立 wrapper checkpoint save/load 拒绝。顶层 `ckpt_path=null`，成员 checkpoint 是两个独立输入，不生成 wrapper 训练 checkpoint。
6. 统一 `src.main` / Hydra 和根 `copmrec_v4_2_inference.sh` 薄入口，只允许单进程单卡（GPU2 → Trainer `[0]`），FP32。复用现有 datamodule、metrics、common writer；主 bundle 保持恰 `keys` / `predictions`，无 trace writer。model 暴露 `pool_contract`：固定 protocol / weights / score_scope / aggregation / branch_encoding / rank_ties / catalog_sha256 / catalog_size / cold_count / top_k，以及 source / control 的 reference、SHA、version、step，control 包含 scale0；实际 Artifact name / digest / role / producer 由配置 metadata 和主审计记录。公共 registry / lineage 必须保留两个 checkpoint，不被同 field_name 覆盖。
7. 唯一一次完整 Evaluation 独立核验原始 22363 用户 / 175 shards 标签、合法 SID、输入/checkpoint/source 字节并复算指标、配对 CI、warm/cold、固定分组与 offender 损益。比较的冻结单模型为 mq8hhof5 source 和 yqsmsdt1 control；二者中最强为 control R10 `.09877923355542638` / N10 `.05042652891207879`。learned-bias 臂不进入 pool，也不替换该增量比较基准。

## Risks / Trade-offs

- raw logits 平均对应 logarithmic opinion pool；单成员正确结果可能被另一成员压低 → 以整体 Recall / NDCG 及 new / lost / shared 分解评估，不把并集命中或错误差异当作融合收益。
- 多成员增加推理成本，且同源成员高度相关 → 单卡串行/同进程实现由工程验证选择，报告实际资源，不追加训练制造多样性。
- 已反复使用 Evaluation 且单 seed / checkpoint 已选择 → CI 是固定 checkpoints 的点态用户配对区间，不声称独立复制或训练 seed 置信度；Testing 保持最终确认。

## Migration Plan

完成聚焦 CPU / Hydra / shell / OpenSpec strict 后，root 通过 Mutagen flush、实际双来源与 runtime source 检查、单卡准备验证，再启动唯一完整 Evaluation；准备检查不计效果证据。旧模型和入口默认行为不变。

Validation 对照固定 LIGER5azn5vm0：R10≥`.09877029021151008`（2209 hits）、N10≥`.0511173919307933`，且相对 LIGER 两项 paired delta CI 下界为正，才进行该唯一 pool 的一次 Testing。Testing 对照042139al，验收R10≥`.07855386128873586`（1757 hits）、N10≥`.04002978116676896`。pool-control 增量 CI 是否为正只决定增量主张强度，不增加总体晋级门槛。

未达到整体双门槛即结题；负向/不确定增量如实保留，不扫描 pair、weight、temperature 或追加训练。Testing 仍全线程最多3次、此前已用1次；本阶段最多消耗其中1次，不重置5 / 30000训练预算。

## Open Questions

固定 pooling 能否提高 control 的 NDCG10 并保持 Recall10，以及能否缓解固定 offender 真目标损失且保留其余用户覆盖，均由唯一一次 Evaluation 回答。未知结果不自动扩大机制或预算。
