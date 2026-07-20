## Context

当前 quantization 模块通过三层继承实现三种模型：

```
LightningModule
└── ResidualQuantization (704行, train_layer_wise flag 分叉)
    └── BaseClusteringModule (267行, 单层量化基类)
        ├── MiniBatchKMeans (147行, K-Means 更新)
        └── VectorQuantization (124行, VQ-STE)
```

`ResidualQuantization` 持有 `nn.ModuleList[BaseClusteringModule]`，通过 `train_layer_wise` 配置在同一类中处理：
- **rkmeans/rvq**（`train_layer_wise=True`）：逐层训练，每层先缓冲数据 → K-Means/initializer 初始化 → 正常梯度训练。
- **rqvae**（`train_layer_wise=False`）：渐进式联合训练，第一层总是训练，后续层在前一层初始化后解锁，同时训练 encoder/decoder/reconstruction。

三层继承的间接性使得追踪一个模型的完整行为需要跨 4 个文件。`BaseClusteringModule` 的 `training_step` / `configure_optimizers` 是死代码（作为子模块时从不被 Lightning 调用）。

## Goals / Non-Goals

**Goals:**
- 将三种模型拆分为独立自包含的类，每个类的完整行为可在单文件内追踪。
- 内联 K-Means / VQ-STE 量化逻辑，消除 BaseClusteringModule 抽象层。
- RQVAE 自包含 K-Means 收敛初始化，消除对 `ClusteringModuleInitializer` 的依赖。
- 清理配置中的 stale keys（`verbose` / `quantization_layer_list` / `evaluator` / `train_layer_wise`）。

**Non-Goals:**
- 不改变三个模型的训练行为和输出（checkpoint state_dict 结构会变，但训练逻辑等价）。
- 不重构 `training_loop_functions.py`（外部工具函数，通过配置注入）。
- 不重构可注入组件（DistanceFunction / STEQuantization / loss functions / initializers / schedulers）。
- 不提供 checkpoint 迁移工具（接受重新训练）。
- 不重构 data pipeline（已在先前变更中完成）。

## Decisions

### 决策 1：三个模型完全独立，无共享基类/mixin

**选择**：每个模型类直接继承 `LightningModule`，不提取共享基类或 mixin。允许代码重复。

**理由**：三个模型共享的代码（eval/val/test/predict、_compute_output_stats、configure_optimizers、metrics 初始化、checkpoint save/load）量不大，且各自有细微差异（RQVAE 多 encoder/decoder/reconstruction）。提取共享基类会重新引入间接性，违背拆分初衷。代码重复优于错误的抽象。

**备选**：提取 `BaseResidualQuantization` mixin 共享 eval/metrics/checkpoint 逻辑。放弃——会重新引入继承层次，且共享接口需要处理 RQVAE 的额外组件，增加复杂度。

### 决策 2：RQVAE 自包含 K-Means 收敛初始化

**选择**：将 `ClusteringModuleInitializer` 的 K-Means 收敛逻辑（KMeansPlusPlus 初始化 → MiniBatchKMeans 迭代到收敛 → centroids 作为 VQ codebook）直接内联到 `ResidualQuantizationVAE` 的初始化流程中。

**理由**：`ClusteringModuleInitializer` 仅 rqvae_train 使用，且其逻辑本质是 RQVAE 的初始化策略。内联后 RQVAE 完全自包含，`ClusteringModuleInitializer` 类可删除。

**备选**：保留 `ClusteringModuleInitializer` 作为独立可注入组件。放弃——它仅服务 RQVAE，且其内部逻辑（创建临时 MiniBatchKMeans、迭代更新）与模型实现强耦合。

### 决策 3：量化逻辑直接内联，不保留 MiniBatchKMeans / VectorQuantization 类

**选择**：删除 `MiniBatchKMeans` / `VectorQuantization` / `BaseClusteringModule` 三个类，将量化逻辑（forward、centroids 更新、初始化缓冲）直接内联到各模型中。每个模型持有自己的 `nn.ParameterList[centroids]`、`init_buffer`、`is_initialized` flags。

**理由**：当前 `BaseClusteringModule` 承担的职责（centroids 管理、init_buffer、初始化编排）在每个模型中实现只需 ~30-50 行，且各模型有差异（K-Means 需 cluster_counts + 手动更新，VQ 需 quantization_strategy + 梯度更新）。保留中间类会阻碍内联。

**备选**：保留 `BaseClusteringModule` 作为 centroids/init_buffer 的工具类。放弃——仍有间接性，且模型需要访问基类的内部状态（centroids、init_buffer），不如直接持有。

### 决策 4：配置 _target_ 直接指向新类，删除 train_layer_wise

**选择**：4 个 model 配置的 `_target_` 从 `ResidualQuantization` 改为对应的新类。删除 `train_layer_wise`、`quantization_layer`（改为内联的量化层参数）、`normalize_residuals`（成为新类的固定行为或构造参数）、以及所有 stale keys。

**理由**：`train_layer_wise` 是分叉三种行为的根源，拆分后不再需要。每个新类的训练策略是固定的（ResidualKMeans/ResidualVectorQuantization = layer-wise，ResidualQuantizationVAE = 渐进式），不需要配置开关。

### 决策 5：launcher_utils dry-run override 调整

**选择**：`launcher_utils.py` 中 dry-run 对 `train_layer_wise=False` 的 override 逻辑（L131-132）删除。新模型的 dry-run 行为由各自实现处理（max_steps=1 时跳过 layer schedule 计算）。

**理由**：`train_layer_wise` 不再是配置参数。dry-run 时各模型需自行处理 max_steps=1 的边界情况。

## Risks / Trade-offs

- **[Checkpoint 不兼容]** 三类新模型的 state_dict 结构与旧 `ResidualQuantization` 不同。→ **缓解**：用户已确认接受重新训练，不需要 backward compatibility。
- **[代码重复]** 三个模型有 ~100 行重复代码（eval/metrics/checkpoint）。→ **缓解**：代码重复是显式的，易于理解和修改；未来如需提取共享逻辑可渐进进行。
- **[实现量大]** 3 个新文件各 ~200-400 行，需要仔细从旧代码中提取并内联。→ **缓解**：按 tasks 分步实现，每步验证 import + config composition。
- **[RQVAE 复杂度]** ResidualQuantizationVAE 是最复杂的模型（encoder/decoder/reconstruction + 渐进式训练 + K-Means 初始化），内联后文件较长。→ **缓解**：单文件可完整追踪行为，优于跨 4 文件追踪。
