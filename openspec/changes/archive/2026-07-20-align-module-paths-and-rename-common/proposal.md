## Why

当前仓库已经将三个实验模块目录 `embedding/`、`quantization/`、`recommendation/` 移动到 `src/` 根下，但代码导入路径与 Hydra 配置中的 `_target_` 仍残留大量旧的 `src.models.*` 引用。同时，`src/models/` 现在只剩 `common/`，目录语义已经与实际内容不匹配。

现在需要先完成路径对齐，再将 `src/models/common` 重命名为更直接的 `src/common`，让实验专有模块与公有模块的边界更清晰。

## What Changes

- 对齐代码与配置中的实验模块引用路径：从旧的 `src.models.embedding|quantization|recommendation.*` 迁移到 `src.embedding|quantization|recommendation.*`。
- 将 `src/models/common` 重命名为 `src/common`，并统一代码与配置中的公共模块引用路径。
- 保守地将当前 `src/models/common` 下内容整体视为公有模块，先统一迁入 `src/common`，不在本轮做更细子域拆分。
- 清理迁移后残留的旧路径引用，确保默认 train/inference 与实验配置均能正常装配。

## Capabilities

### New Capabilities
- `module-path-alignment`: 统一实验模块与公有模块的代码/配置路径，使仓库目录结构与导入语义一致。

### Modified Capabilities

## Impact

- 受影响代码：`src/embedding/`、`src/quantization/`、`src/recommendation/`、`src/models/common/` 及引用它们的 `src/utils/` / 其他模块
- 受影响配置：`configs/experiment/*.yaml` 中的 `_target_` 路径
- 不涉及新依赖，但会涉及较大范围的 import 和配置字符串替换
