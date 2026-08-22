## Context

RQ-VAE 当前训练与验证日志混合了不同目标。训练阶段的 `loss` 是 quantization loss 和 reconstruction loss 的加权和；验证/测试阶段的 `loss` 只来自 encoded embedding 的 residual quantization loss。当前 residual ratio / mse 也使用 encoded residual，却用原始 input embedding norm 归一化，导致指标空间不明确。

这会直接影响后续调参判断：reconstruction loss 下降可能掩盖 quantization loss 上升，validation 曲线也无法和 train total loss 对齐。因此本次变更先修可观测性和指标契约，不改变 RQ-VAE 的训练阶段策略。

## Goals / Non-Goals

**Goals:**

- 让 RQ-VAE train、validation、test 的 `loss` 使用同一个加权 total loss 语义。
- 让 RQ-VAE validation/test 同时暴露 `quantization_loss` 和 `reconstruction_loss`。
- 将 RQ-VAE encoded-space residual 指标与 reconstruction-space 指标拆开命名。
- 更新 RQ-VAE metric config，让 W&B 曲线能直接比较 total、quantization、reconstruction 目标。
- 用小型 CPU 单测覆盖 eval payload 和指标空间语义。

**Non-Goals:**

- 不引入 reconstruction warmup、逐层 codebook 训练阶段机或 joint finetune。
- 不调整 RQ-VAE optimizer、learning rate、loss weight 或 encoder/decoder 结构。
- 不改变 RKMeans/RVQ 的 metric payload。
- 不重跑实验或写入新的实验结果。

## Decisions

1. **先修 eval payload，而不是先调训练策略**

   RQ-VAE 现在缺少可靠反馈信号。若先调 optimizer 或阶段策略，无法判断改善来自 codebook、reconstruction，还是日志口径变化。先对齐 payload 可以让后续实验有稳定观测面。

2. **`loss` 在所有阶段都表示 weighted total loss**

   训练优化目标本身是加权总损失，因此 `val/loss` 和 `test/loss` 也应计算同一公式。quantization 和 reconstruction 分量通过独立字段记录，而不是让 `loss` 在不同阶段代表不同含义。

3. **RQ-VAE 指标空间显式命名**

   encoded residual 指标使用 encoded embedding 作为 norm denominator，并用 `encoded_*` 命名。decoder reconstruction 指标比较 reconstructed embedding 与 normalized input embedding，并用 `reconstruction_*` 命名。这样避免把 64 维 encoded residual 与 768 维原始输入空间混合解释。

4. **保留旧通用字段时必须明确迁移**

   RQ-VAE 可以在过渡期保留通用 `mse` / residual ratio 字段以减少配置冲击，但新判断应依赖空间明确的新字段。如果实现选择直接替换旧字段，必须同步更新 config、tests 和文档，避免 W&B 曲线名误导。

## Risks / Trade-offs

- **Risk:** RQ-VAE 新曲线名会打断与旧 W&B run 的直接字段对比。
  **Mitigation:** 在文档和 config 中保留 total、quantization、reconstruction 三类核心曲线，并说明旧 run 的 `val/loss` 不是 total loss。

- **Risk:** validation/test 中计算 decoder reconstruction 会增加少量开销。
  **Mitigation:** RQ-VAE validation 已在较低频率运行；该开销换来 train/val 语义一致，优先级更高。

- **Risk:** 如果 BatchNorm 在 validation 计算路径中状态不一致，reconstruction metric 可能和 train 略有分布差异。
  **Mitigation:** 保持 Lightning 标准 train/eval mode，不为指标引入额外模式切换；只比较趋势，不把单点差异解释为训练失败。
