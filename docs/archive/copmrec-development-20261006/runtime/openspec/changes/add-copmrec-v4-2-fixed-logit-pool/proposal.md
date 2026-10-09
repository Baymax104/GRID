## Why

已完成的 source v4 best6000 与 frozen-bias control best6000 在同一完整 Validation 上有 320 个新增、258 个丢失命中，两个固定用户组均呈双向差异；control 改善 Top10 却降低 Top5，并损害固定 639 个 offender 真目标用户。该当前配对证据支持一次固定等权全目录 logit pooling 验证，检验能否保留两者的推荐优势，不推定融合有效或承诺 10%。

原 v4 与 v4.1 阶段均已关闭，累计 5 次训练 / 30000 steps 已用尽；本变更仅允许 0 训练、一次固定单卡完整 Evaluation，条件达标后最多一次 Testing，累计 Testing 上限保持 3 次、已用 1 次。完整五项门禁见 `docs/copmrec-v4-2-fixed-logit-pool-research.md`。

## What Changes

- 新增仅推理的 v4.2 固定 logit pool，来源唯一为 `nj9elah1` v4 best6000 与 `l3zyr91b` v4.1 frozen-bias control best6000；冻结 URI、原始 SHA、结构与来源契约。
- 每个成员各自编码同一真实历史，计算自己的完整目录 dense logits；逐目录行执行固定 `0.5 * source_logits + 0.5 * control_logits`，再统一稳定取 Top10 完整 SID。
- 复用公共 checkpoint loader、统一 `src.main` / Hydra、标准 ModelOutput 和 common writer，保持单卡与来源记录；不新增训练、独立分析入口或 Top10 分数拼接。
- 添加最小 CPU 契约测试、薄推理配置与根脚本，完成来源、目录、模式及单进程防护。
- 仅执行一次完整 Evaluation 并独立配对复算；只有同 split LIGER 两项均达 +10% 才允许固定 pool 的一次 Testing。无 weight / temperature / pair / checkpoint / bias / LR 扫描。

## Capabilities

### New Capabilities

- `copmrec-fixed-logit-pool`：两个固定已训练模型的全目录等权 logit pooling、严格目录与来源、单卡统一推理以及有界验证。

### Modified Capabilities

无。既有 v0、v4、v4.1 的行为与入口不改。

## Impact

新增 recommendation 推理模块、model / experiment 配置、根推理脚本及聚焦测试；复用现有 artifact、lineage、metric 与 writer，不引入依赖。推理需运行两个已有成员，增加编码与模型存储成本。机制按 Heskes 的 logarithmic opinion pool 定位，性能由真实 Recall / NDCG 判定，不将概率平均论文、错误互补或曝光下降作为效果保证。
