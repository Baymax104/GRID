## Why

`src/common/modules` 中剩余模块已经不再是真正共享模块：`EmbeddingAggregator` 只服务 semantic embedding，`MLP` 与 `NormalizeLayer` 只服务 RQVAE。继续放在 common 命名空间会模糊模块归属，也与近期将推理、metric、data helper 收敛到领域目录的方向不一致。

本变更将这些模块移动到实际消费域，并删除空的 `src/common/modules`，让 common 命名空间只保留确实跨域复用的组件。

## What Changes

- 将 `EmbeddingAggregator` 从 `src/common/modules/embedding_aggregator.py` 移到 embedding 领域目录。
- 将 `MLP` 与 `NormalizeLayer` 从 `src/common/modules/` 移到 RQVAE 领域目录。
- 更新 Python imports 和 Hydra `_target_` 配置。
- 删除空的 `src/common/modules` 包。
- 更新 OpenSpec living specs，明确 common modules 不再作为当前模块归属目标。
- 增加或更新 focused tests / residual scans，防止 `src.common.modules` 路径残留。

## Capabilities

### New Capabilities

### Modified Capabilities
- `module-path-alignment`: common/shared module 路径规则调整为只保留真实共享组件；实验专属模块 MUST 使用对应 `src.embedding.*` 或 `src.quantization.*` 路径。
- `flattened-component-contract`: `src/common/modules` SHALL be removed after remaining pseudo-shared modules move to their consuming domains.

## Impact

- 影响代码：`src/common/modules/`、`src/embedding/`、`src/quantization/rqvae/`。
- 影响配置：`configs/model/sem_embeds_inference.yaml`、`configs/model/rqvae_train.yaml`。
- 影响测试：模型配置路径断言或新增 focused import/instantiate tests。
- 不新增依赖，不改变 `EmbeddingAggregator`、`MLP`、`NormalizeLayer` 的运行行为。
