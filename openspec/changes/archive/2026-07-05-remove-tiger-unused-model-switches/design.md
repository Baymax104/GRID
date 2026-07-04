## Context

当前仓库中，`compile` 只作为 `TransformerBaseModule` 的构造参数与 `self.hparams` 一部分存在，唯一相关实现是一段已注释掉的 `torch.compile(...)` 代码，因此它对实际训练/推理行为没有任何影响。`weight_tying` 则在基类 `get_embedding_table()` 中保留了一个二选一分支，用于决定评估阶段 retrieval evaluator 的 key embeddings 来源；但当前官方 TIGER 实现已经在子类中定义了自己的 embedding table 路径，而且官方 experiment 始终固定 `weight_tying: true`，并没有把它作为真实的模型变体开关使用。

本次目标不是引入新的 embedding 语义，而是删除配置层上没有实际选择价值的开关，并将代码固定到当前官方默认行为。

## Goals / Non-Goals

**Goals:**
- 删除 `compile` 配置与死代码入口。
- 删除 `weight_tying` 配置与基类中的可选分支。
- 保持当前官方 TIGER experiment 的实际运行语义不变。
- 让基类评估路径显式固定使用 encoder input embeddings。

**Non-Goals:**
- 不启用 `torch.compile`。
- 不重构 TIGER 子类中的 embedding table 设计。
- 不引入新的模型配置开关来替代这两个字段。

## Decisions

### 1. 删除 `compile`，不再保留预留接口
- 决策：从配置与 `TransformerBaseModule` 构造参数中完全移除 `compile`。
- 原因：当前没有任何实际执行路径会消费它，继续保留只会制造“似乎支持编译优化”的错误预期。

### 2. 删除 `weight_tying`，固定为当前默认语义
- 决策：移除 `weight_tying` 配置，并将 `TransformerBaseModule.get_embedding_table()` 固定返回 `self.encoder.get_input_embeddings().weight`。
- 原因：这与当前官方 experiment 固定使用 `weight_tying: true` 的默认行为一致，同时避免继续暴露未真正使用的可选分支。

### 3. 不对 TIGER 子类的自定义 embedding table 路径做结构性重构
- 决策：保留 `SemanticIDEncoderDecoder` 现有的 `get_embedding_table(table_name, hierarchy)` 设计，只清理已经失去配置意义的通用开关。
- 原因：本次目标是配置接口收敛，而不是推荐模型内部架构重写。

## Risks / Trade-offs

- [少量外部私有配置可能仍传入这两个字段] → 会失去兼容性，但这是用户接受的 breaking change。
- [基类固定 encoder embeddings 后，未来若真想切换语义需要重新设计] → 这是可接受的，届时应通过新的显式设计而不是保留死开关。
- [名称上仍可能让人误以为 TIGER 子类完全依赖基类 `get_embedding_table()`] → 本次不重构子类结构，只保证官方配置接口更诚实。

## Migration Plan

1. 删除 TIGER experiment 中的 `weight_tying` / `compile`。
2. 删除 `TransformerBaseModule` 中的对应 init 参数和 compile 注释残留。
3. 将基类 `get_embedding_table()` 固定到 encoder input embeddings。
4. 做最小静态检查与全文搜索，确认仓库中不再保留这两个官方配置入口。

## Open Questions

- 当前无阻塞性开放问题。
