## Context

`src/common/modules` 当前只剩三个模块：

- `EmbeddingAggregator`：由 `src/embedding/hf_language_model.py` 和 `configs/model/sem_embeds_inference.yaml` 使用。
- `MLP`：由 `configs/model/rqvae_train.yaml` 的 encoder/decoder 使用。
- `NormalizeLayer`：由 `configs/model/rqvae_train.yaml` 的 preprocessing layer 使用。

这些模块没有跨 embedding、quantization、recommendation 的真实共享关系。继续放在 `src.common.modules` 会让“common”成为历史路径，而不是领域边界。近期已将 inference output、metrics、data helpers 迁移到更明确的归属目录，本变更延续同一结构原则。

## Goals / Non-Goals

**Goals:**

- 将 embedding-only 的 `EmbeddingAggregator` 移入 `src/embedding`。
- 将 RQVAE-only 的 `MLP` 和 `NormalizeLayer` 移入 `src/quantization/rqvae`。
- 更新所有 Python imports 与 Hydra `_target_`。
- 删除 `src/common/modules` 包。
- 更新 living specs，避免继续要求实验专属模块使用 `src.common.modules.*`。

**Non-Goals:**

- 不改变 `EmbeddingAggregator`、`MLP`、`NormalizeLayer` 的行为。
- 不重构 RQVAE encoder/decoder 架构。
- 不迁移 `src/common/components/loss_functions.py` 或 `scheduler.py`。
- 不新增外部依赖。

## Decisions

### EmbeddingAggregator 移入 embedding 领域

`EmbeddingAggregator` 只服务 semantic embedding 模型，因此迁移到 `src/embedding/embedding_aggregator.py`。`src/embedding/hf_language_model.py` 直接从 embedding 本地模块导入，`sem_embeds_inference` 配置也使用 `src.embedding.*` 路径。

理由：

- 聚合逻辑与 HF language model embedding 输出强相关。
- 它不是 quantization 或 recommendation 可复用模块。
- 配置入口与代码入口保持同一领域命名空间。

### MLP 与 NormalizeLayer 移入 RQVAE 领域

`MLP` 与 `NormalizeLayer` 只由 RQVAE 配置实例化，因此迁移到 `src/quantization/rqvae/`。

理由：

- 当前没有 RKMeans、RVQ、TIGER 或 embedding 路径引用这些模块。
- RQVAE 的 encoder/decoder 配置应就近引用 RQVAE 组件。
- 避免 common 命名空间承载单实验实现细节。

### 删除 src/common/modules

迁移后 `src/common/modules` 不再保留空包或兼容 re-export。

理由：

- 项目已有 residual scan 与配置 `_target_` 校验习惯，直接删除比保留兼容层更能暴露旧路径残留。
- 现有代码没有外部公共 API 兼容要求。

## Risks / Trade-offs

- [Risk] 外部脚本可能仍引用 `src.common.modules.*`。  
  Mitigation: 官方代码和配置全部迁移；内部项目不保留兼容 re-export，旧路径应尽早失败。

- [Risk] Hydra `_target_` 路径遗漏会在运行时失败。  
  Mitigation: 对相关 model configs 做 residual scan，并运行 focused config/model tests。

- [Risk] living spec 仍要求 `src.common.modules.*` 会与实现冲突。  
  Mitigation: 同步修改 `module-path-alignment` 与 `flattened-component-contract`。

## Migration Plan

1. 移动 `EmbeddingAggregator` 到 `src/embedding/embedding_aggregator.py`，更新 import 和 `sem_embeds_inference` 配置。
2. 移动 `MLP`、`NormalizeLayer` 到 `src/quantization/rqvae/`，更新 `rqvae_train` 配置。
3. 删除 `src/common/modules` 包。
4. 更新 OpenSpec living specs。
5. 运行 residual scan：`src.common.modules`、`common/modules`、旧 `_target_` 必须清零。
6. 运行 focused tests、scoped ruff、OpenSpec strict validation。

## Open Questions

无。
