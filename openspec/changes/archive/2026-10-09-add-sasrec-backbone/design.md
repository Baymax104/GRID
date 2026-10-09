## Context

依据作者 [model.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py) 和 [modules.py](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py)。官方 attention 没有 output projection；Q 输入为 LN(x)，K/V 输入为 x，residual 加 LN(x)。FFN 接收 LN(attention output)，两层宽度均为 hidden size，带 ReLU/dropout，residual 加 FFN 输入。每个 block 后清零 padding，最后再 LN。

## Goals / Non-Goals

**Goals:** 实现官方计算、mask、共享 embedding、损失与固定来源记录，独立核验前向数值和梯度路径。

**Non-Goals:** 本模块不装配数据、Lightning、运行命令或正式实验。后续按 data、training/evaluation、pipeline 三个模块分别提案实施。

## Decisions

- 显式 Q/K/V 线性投影和分头计算；不使用带额外 output projection 的 MultiheadAttention。
- 位置 embedding 使用固定槽位 `0..L-1`，零商品 embedding 加位置后由商品 mask 清零。输入按官方左 padding，取最后槽位评分。
- key/query mask 按官方输入向量绝对值求和的非零性计算。mask logits 使用官方有限常数 `-2**32+1`，避免全 mask softmax NaN；attention 权重 dropout，FFN 激活后与第二层后 dropout。
- item/position embedding 和线性权重使用 Xavier uniform，bias 为零，LN epsilon 为 `1e-8`。TensorFlow 与 PyTorch RNG 不作跨框架随机数一致承诺。
- 官方 BCE 包含 `1e-24`，直接保留概率公式与有效正标签平均；L2 为 `0.5*l2_emb*(raw_item_weight²+position_weight²)`，与 tf.contrib.layers.l2_regularizer 对应。有效 padding embedding 恒零，raw 第零行仅参与 L2。
- 以独立 NumPy 转写官方公式的前向参考、未来扰动不变性、手算 BCE/L2 与梯度测试核验。参考只作测试，不新增运行入口或 TensorFlow 依赖。

## Risks / Trade-offs

- 框架差异 → 固定参数下核验公式数值；不宣称逐比特 TensorFlow 复现。
- 普通 Transformer 的 norm/residual 与官方不同 → 仅实现官方一种结构。
- 空历史或无有效标签 → 明确报错；后续数据模块负责过滤无法训练的短序列。
