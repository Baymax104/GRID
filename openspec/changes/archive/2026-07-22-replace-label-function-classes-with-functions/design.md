## Context

`collate_fn_train` 当前通过 label function 类实例间接调用 label generator。`NextKTokenMasking` 和 `Identity` 都只是无外部生命周期的 tensor 转换逻辑，却被包装为类并继承 `LabelFunction` 抽象基类。Hydra 配置也因此需要额外的 `transform:` 层，增加了配置形状和运行时协议的复杂度。

当前实际使用者只有 TIGER train/eval 的 `sequence_data` label 配置，使用 `NextKTokenMasking(next_k=${num_hierarchies})`。因此可以直接迁移为 Hydra `_partial_` 纯函数 callable，并用 `GeneratedLabels` 表达生成后的 masked input 与 target semantic IDs。

## Goals / Non-Goals

**Goals:**

- 删除不必要的 label function 类层级和抽象基类。
- 将 label generator 暴露为纯函数 callable。
- 简化 `collate_fn_train` 对 label callable 的调用方式。
- 保持现有 `next_k` masking 行为、输出 shape 和 TIGER loss 输入不变。

**Non-Goals:**

- 不改变 TIGER label masking 语义。
- 不改变 `SequentialModuleLabelData` / `SequentialModelInputData` 的字段结构。
- 不改变 TIGER 当前“最后 `num_hierarchies` 个 token 作为 labels”的训练目标。
- 不新增新的 label generation 算法。

## Decisions

### D1: 使用纯函数而不是 callable class

将 `NextKTokenMasking.transform_label` 迁移为：

```python
def next_k_token_masking(
    sequence: torch.Tensor,
    padding_token: int,
    masking_token: int,
    next_k: int = 5,
) -> GeneratedLabels:
    ...
```

理由：`next_k` 是唯一参数，可由 Hydra `_partial_: true` 绑定；不需要对象状态或继承接口。

### D2: `collate_fn_train` 直接调用 label callable

`label_generate_functions[field_name]` SHALL 是可调用对象，`collate_fn_train` 直接调用：

```python
label_function_output = label_function(
    sequence=current_sequence,
    padding_token=padding_token,
    masking_token=masking_token,
)
```

这使配置中的 label callable 与 collate runtime 消费方式一致。

### D3: 使用 `GeneratedLabels`

`GeneratedLabels` 作为统一返回协议，包含 `input_ids` 与 `target_ids`，避免继续暴露旧的 generic label output 字段。

### D4: 不保留旧类 alias

不为 `NextKTokenMasking` / `Identity` 保留兼容 alias。旧类名会让配置继续选择类协议，削弱本次变更目标。仓库内配置一次性迁移到纯函数 target。

## Risks / Trade-offs

- **旧配置无法 instantiate** → 这是预期 BREAKING 行为；通过 grep 确认官方配置中不再引用旧类名。
- **label shape 变化影响 TIGER loss** → 用 collate smoke 验证 `labels.shape == (batch_size * next_k,)`、`label_location.shape == (batch_size * next_k, 2)`。
- **Hydra `_partial_` 配置写法错误** → 对 `tiger_train` compose / datamodule instantiate 执行 smoke。
