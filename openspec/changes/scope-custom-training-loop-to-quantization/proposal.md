## Why

当前仓库通过 `training_loop_function` 这个通用可注入 hook 暴露手动优化能力，但从实际使用面看，这套机制只服务于 quantization 训练阶段的特殊初始化逻辑（如 DDP 下对初始化 loss 做 `world_size` 缩放、使用临时 SGD 初始化步、手动 step scheduler）。

然而该机制目前同时暴露在：

- `src/quantization/residual_quantization.py`
- `src/common/modules/transformer_base_module.py`

后者是更广义的 transformer / recommendation 通用基类，但当前并没有任何 recommendation / transformer 训练实验真正使用这个 hook。这会制造错误暗示：仿佛“任意模型都应通过 `training_loop_function` 自定义 train loop”，而实际上它只是 quantization 子域的特殊异常路径。

此外，相关实现文件 `src/common/components/training_loop_functions.py` 也被归类在 `common` 下，但其内容与量化初始化强绑定，并不属于真正的项目级公共能力。

为了让模块边界更清晰，需要把这套机制收敛到 quantization 域，避免在通用基类里暴露与主干训练语义无关的高复杂度手动优化入口。

## What Changes

- **BREAKING（内部架构收敛）** 从 `TransformerBaseModule` 移除 `training_loop_function` 参数与相关手动优化分支
- 将初始化专用训练函数从 `src/common/components/training_loop_functions.py` 下沉到 quantization 更贴近的命名空间
- 清理 quantization 相关配置中对该函数的引用路径，使其不再表现为“项目级 common hook”
- 清理与训练无关却仍暴露该配置的模型/实验（例如 `rkmeans_inference`）中的无意义配置项
- 在 `ResidualQuantization` 侧保留最小必要的特殊训练路径，使 quantization 训练仍能完成初始化阶段的 DDP 特殊优化

## Capabilities

### New Capabilities
- `quantization-initialization-training-strategy`: 规定 quantization 初始化阶段的特殊手动优化路径应局部封装在 quantization 子域内，而不是暴露为全项目通用 hook

### Modified Capabilities
- `transformer-base-training-contract`: 通用 transformer 训练基类恢复为标准 Lightning automatic optimization 契约，不再暴露 quantization 特有的 train loop 注入接口

## Impact

- 受影响代码：
  - `src/common/modules/transformer_base_module.py`
  - `src/common/components/training_loop_functions.py`（移动/重命名）
  - `src/quantization/residual_quantization.py`
- 受影响配置：
  - `configs/model/rkmeans_train.yaml`
  - `configs/model/rqvae_train.yaml`
  - `configs/model/rvq_train.yaml`
  - `configs/model/rkmeans_inference.yaml`
- 不改变 `tiger_train` 等 recommendation 主链训练语义；主要是架构边界收敛与无效暴露清理
