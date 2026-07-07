## Context

`training_loop_function` 当前只在 quantization 相关配置里实际启用：

- `rkmeans_train`
- `rqvae_train`
- `rvq_train`
- 以及形式上出现在 `rkmeans_inference` 配置中（但 inference 不走 train loop）

其具体实现 `scale_loss_by_world_size_for_initialization_training_loop()` 的语义也非常专用：

- 仅用于初始化阶段
- 依赖 `is_initialized`
- 在 DDP 下对 loss 做 `world_size` 缩放
- 使用临时 SGD 初始化步
- 在 manual optimization 下补手动 `scheduler.step()`

这不是一个“项目通用训练扩展点”，而是量化初始化的局部策略。当前同时把它暴露给 `TransformerBaseModule`，会使通用基类接口变宽，并把 manual optimization 的复杂度泄漏到 recommendation 主链的抽象层。

## Goals / Non-Goals

**Goals:**
- 将 custom training loop 能力收敛到 quantization 子域
- 恢复 `TransformerBaseModule` 的标准 automatic optimization 契约
- 让代码结构与真实使用面一致：特殊初始化策略属于 quantization，而不是 common transformer training
- 清理 inference 场景里无意义的 `training_loop_function` 暴露

**Non-Goals:**
- 本轮不重写 quantization 初始化算法本身
- 本轮不强制把 `ResidualQuantization` 内的策略从 callable 立刻改成 bool/enum，如果会扩大改动面可先保留 callable 形式
- 不改变 `tiger_train` / `tiger_inference` 的训练/推理行为

## Decisions

### D1: 从 `TransformerBaseModule` 移除 `training_loop_function`
- **选择**：删除 `TransformerBaseModule.__init__` 中的 `training_loop_function` 参数、`automatic_optimization=False` 分支，以及 `training_step()` 中对 hook 的调用。
- **理由**：当前无 recommendation / transformer 模型使用该机制；继续保留只会制造错误抽象。
- **备选**：保留但标记 deprecated——否决，延长无效接口寿命而无现实收益。

### D2: 初始化专用训练函数下沉到 quantization 子域
- **选择**：将 `src/common/components/training_loop_functions.py` 中的初始化专用逻辑迁移到 quantization 更贴近的命名空间（例如 `src/quantization/training_strategies.py` 或 `src/quantization/training_loop_functions.py`）。
- **理由**：实现本身与 quantization 初始化强绑定，不应再以 `common` 面貌暴露。

### D3: `ResidualQuantization` 保留最小必要的手动优化入口
- **选择**：本轮先保留 `ResidualQuantization` 侧的特殊训练入口，但只在 quantization 域内可见与配置。
- **理由**：这样能先完成边界收敛，而不一次性把内部训练策略表达也重构掉，降低风险。
- **后续机会**：若下一轮继续收敛，可再把 callable 改成更具体的策略名或内部显式分支。

### D4: 清理 `rkmeans_inference` 中无意义的训练 loop 配置
- **选择**：移除 `configs/model/rkmeans_inference.yaml` 中对 `training_loop_function` 的暴露。
- **理由**：inference 不走 `training_step`，保留该配置只会让人误判其参与运行。

## Risks / Trade-offs

- **[风险] 未来 transformer 模型也可能需要 manual optimization** → 可接受。等未来出现真实需求时，再以更明确的语义新增，而不是今天为假想需求保留过宽接口。
- **[风险] quantization 配置路径变动会影响已有 experiment** → 缓解：统一修改相关 `configs/model/*.yaml` 引用，并做 compose/import 最小验证。
- **[权衡] `ResidualQuantization` 仍暂时保留 callable 形式，未做到最彻底收敛** → 可接受，本轮目标先是“只在 quantization 域暴露”，而不是一次性重塑全部内部策略表达。

## Migration Plan

1. 从 `TransformerBaseModule` 删除 `training_loop_function` 支持
2. 将初始化专用训练函数迁移到 quantization 子域
3. 更新 quantization train 配置引用路径
4. 删除 `rkmeans_inference` 中无效的 `training_loop_function` 配置
5. 验证 recommendation 主链仍走标准 automatic optimization，quantization 训练仍可保留特殊初始化路径
