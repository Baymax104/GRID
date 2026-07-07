## 1. 收窄通用基类接口

- [x] 1.1 从 `src/common/modules/transformer_base_module.py` 的 `__init__` 中移除 `training_loop_function` 参数
- [x] 1.2 删除 `TransformerBaseModule` 中基于 `training_loop_function` 的 `automatic_optimization = False` 分支
- [x] 1.3 删除 `TransformerBaseModule.training_step()` 中对 `training_loop_function(...)` 的调用，恢复纯标准 Lightning automatic optimization 流程

## 2. 将特殊训练策略下沉到 quantization 域

- [x] 2.1 将 `src/common/components/training_loop_functions.py` 迁移/重命名到 quantization 更贴近的模块路径
- [x] 2.2 更新 `ResidualQuantization` 及相关配置，使其引用新的 quantization 域策略路径
- [x] 2.3 确认 `ResidualQuantization` 仍能在初始化阶段保留 manual optimization 能力

## 3. 清理配置暴露

- [x] 3.1 更新 `configs/model/rkmeans_train.yaml`、`rqvae_train.yaml`、`rvq_train.yaml` 中 `training_loop_function` 的 `_target_` 路径
- [x] 3.2 删除 `configs/model/rkmeans_inference.yaml` 中无意义的 `training_loop_function` 暴露

## 4. 验证收尾

- [x] 4.1 最小验证 recommendation 主链（如 `tiger_train` 配置）不再暴露 `training_loop_function`
- [x] 4.2 最小验证 quantization train 配置仍可 compose 且引用到新的 quantization 域策略路径
- [x] 4.3 复查仓库，确认 `training_loop_function` 不再出现在 `TransformerBaseModule` 与 recommendation 主链配置中
