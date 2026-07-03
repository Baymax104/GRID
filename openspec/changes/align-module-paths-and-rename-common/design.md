## Context

当前仓库处于模块迁移的中间态：实验模块 `embedding/`、`quantization/`、`recommendation/` 已移动到 `src/` 根下，但代码 import 与 Hydra 配置中的 `_target_` 仍残留旧的 `src.models.*` 路径。同时 `src/models/` 已不再承载实验模块，仅剩 `common/`，目录命名与实际职责不再匹配。

探索结果显示，`src/models/common/` 下的大部分内容都可作为当前阶段的公有模块整体迁入 `src/common/`。其中某些模块（如 `eval_metrics`、`transformer_base_module`、`quantization_strategies`、`training_loop_functions`）更偏“子域公共”，但本轮不继续做二次分层。

## Goals / Non-Goals

**Goals:**
- 统一实验模块代码导入与配置 `_target_` 到新的 `src.embedding|quantization|recommendation.*` 路径。
- 将 `src/models/common` 重命名为 `src/common`。
- 统一代码与配置中对公有模块的引用到 `src.common.*`。
- 保证默认训练/推理与实验配置在迁移后仍可装配。

**Non-Goals:**
- 不在本轮对 `src/common` 再做按子域拆分。
- 不重构模块内部实现逻辑，只做路径与目录语义对齐。
- 不引入新的公共抽象边界设计。

## Decisions

### 1. 分两类路径统一处理
- 决策：将迁移分为两类路径替换：
  1. 实验模块路径：`src.models.embedding|quantization|recommendation.*` → `src.embedding|quantization|recommendation.*`
  2. 公有模块路径：`src.models.common.*` → `src.common.*`
- 原因：这与当前目录现实一致，也最容易系统性验证。

### 2. `src/models/common` 整体迁入 `src/common`
- 决策：本轮直接将 `src/models/common/components` 与 `src/models/common/modules` 整体迁入 `src/common/`。
- 原因：当前 `models` 目录只剩公有内容，继续保留 `models/common` 只会增加过渡复杂度。
- 备选方案：保留 `src/models/common` 不动，仅修实验模块路径。未采用，因为会让目录语义继续混乱。

### 3. 子域公共模块先不细拆
- 决策：即便某些模块更偏量化或推荐子域，也先统一放入 `src/common/`。
- 原因：降低本轮 blast radius，优先完成路径对齐与目录收敛。

## Risks / Trade-offs

- [替换范围大，容易漏改配置路径] → 代码与 YAML 路径分开扫描，并在实现后做全文搜索验证。
- [存在历史注释/文档残留旧路径] → 本轮优先保证代码与配置可运行，注释残留可作为次级清理项。
- [某些“公共模块”未来还要再拆] → 先接受 `src/common` 作为过渡收敛点，后续如有需要再演进。

## Migration Plan

1. 先替换代码中的实验模块旧路径引用。
2. 再替换配置里的 `_target_` 实验模块路径。
3. 迁移 `src/models/common` 到 `src/common`。
4. 统一替换代码与配置中的 `src.models.common.*` 路径。
5. 做全文搜索与最小静态检查，确认无旧路径残留。

## Open Questions

- 当前无阻塞性开放问题。
