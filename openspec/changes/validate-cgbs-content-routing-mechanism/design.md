## Context

Beauty 内容初始化 A 的训练 run 为 `5g3wpbg7`。当前 Full 不调用 `_initialize_tokens`，Hybrid 包含候选并集，均不能充当 A 上增加一个组件的嵌套对照。已观察到局部 rank 改善却没有转化为真实 beam 存活，因此需要逐用户连接评分、路径与结果。

## Goals / Non-Goals

**Goals:** 补 B/C 两个训练条件，并使固定 checkpoint 可执行有身份记录的推理干预。默认 Beauty、seed42、20k steps、每设备 batch128、物理 GPU0/1、从头训练；A 仅在数据、上游产物与训练协议一致时复用。

**Non-Goals:** 不引入新方法、门控、损失或原型结构；不自动启动 GPU 实验；不以单种子开发结果宣称论文结论成立；不改历史八条件。

## Decisions

1. `content_init_aux` 同 A 初始化，并用相同的 content_query + item CE；生成仅合法 token 概率。`content_init_full` 同 A 初始化，其他行为同 Full。额外模块初始化采用局部 RNG，保证 backbone 与后续随机流一致。两个 arm 不互相加载 checkpoint。
2. 推理 `content_scoring=trained|off|exact|shuffled_exact|rerank` 与训练契约分离。`trained` 完全保持历史路径；干预写入 resolved config 和 prefix trace metadata。非默认模式不得进入 training_step。关闭分支仍使用合法 token 概率。
3. 精确质量是 descendant item 的 logsumexp，不是 oracle 或效用上界。分块计算，保持候选合法性；`shuffled_exact` 使用固定 CPU RNG 置乱 feature→item 关系，SID 与分支大小不变。精确/置乱对照使用相同 alpha：C 默认复用每层训练 alpha；B 默认固定 alpha_initial=0.1，不作调参。置乱精确评分应与精确评分比较，不能将差异都归因于原型或内容。
4. `rerank` 仅重排 token-only beam 的相同完整候选，不作 dense union；使用归一化 item 内容概率与 token 路径概率混合，C 的 alpha 使用层均值，B 使用固定 alpha_initial。其干预包含评分公式变化，只是实用后重排对照；trace 禁用，避免把重排后的序位误标为 beam 轨迹。相同候选的 Hit@K 不应改变。
5. 所有预测复用 `Trainer.predict` 和共享 recommendation/prefix trace writer。相同目标前缀的 teacher-forcing trace 可对齐比较；真实 beam trace 必须独立报告，不把标签传入评分。生成在有无标签观测时必须相同。
6. 冻结训练条件，训练入口默认仅 B/C 顺序执行，GPU0/1；额外 override 置后并记入 W&B。固定条件被 override 改变时，该 run 不得并入主对比。后续推理要求显式 best checkpoint 引用，不用不确定的 `*.ckpt` 或 latest 推断。

## Risks / Trade-offs

- 历史开发集已被反复观察 → 本轮仅机制筛选；通过后才做独立最终评估和多 seed 确认。
- C 关闭分支存在共适应影响 → 不能代替 B/C 独立训练对照。
- B 的精确干预没有联合训练 → 阴性仅限制此固定方案，不否定所有内容路线。
- 精确评分额外计算昂贵 → 分块降低显存，记录成本；禁止当作现有压缩分支的效率结果。
- 置乱只证明固定模型依赖内容对应关系 → 不夸大为训练因果分离。
- 实际瓶颈尚未确认 → 判定规则先固定，不通过就结束本机制或限定为具体近似问题；不得自动扩展搜索。
