## Why

已有实验未支持仅依靠解码配额或轻量频率重加权即可稳定改善 Tail 的假设。下一步需要在训练中引入可共享的 item 内容证据，并在 SID 分支决策中直接使用该证据，形成可以被完整对照和消融检验的方法。研究设计见 `E:/projects/research/ideas/2026-09-12-main-method-design.md`；本变更实现其中的 Catalog-Grounded Branch Scoring（CGBS）。

## What Changes

- 新增独立 TIGER CGBS 模块，复用原有 encoder、decoder 和数据协议，不修改原 TIGER 主数据流。
- 按 item key 对齐语义向量与完整 SID；构建固定 PCA 内容表示及含 item 数量的多原型前缀索引，并在 checkpoint 保存和校验来源。
- 实现共享内容 query、合法分支概率混合、全目录辅助训练和相同评分下的 beam search。
- 实现 original、mask_ce、token_content_init、single_prototype、full、no_aux、shuffled、hybrid 八个明确命名的实验条件。Hybrid 是生成与稠密检索的对照，不宣称复现完整 LIGER。
- 新增薄 experiment 配置、训练与推理启动脚本、两组 GPU 实验队列和实施说明；完整实验由用户手动启动。
- 使用聚焦单元测试、Hydra compose、脚本参数测试和 OpenSpec strict 校验验收。

## Capabilities

### New Capabilities

- `tiger-catalog-grounded-scoring`: 固定目录内容索引、内容分支混合训练与推理、消融控制、checkpoint 来源检查及实验入口。

### Modified Capabilities

无。既有 TIGER、输出 bundle、artifact 解析和主入口契约继续复用。

## Impact

- 新文件主要位于 `src/recommendation/tiger_catalog_grounded/`、`src/data/components/`、对应 configs、tests 和根目录 shell 启动脚本。
- 不增加依赖，不修改已有 baseline 默认行为，不启动完整实验，不提交或推送代码。
- 成本增加来自固定原型索引、训练时全目录辅助 CE 和 Hybrid 对照的稠密检索；必须记录实际 GPU 时间与峰值显存，不能由单元测试声称方法有效。
