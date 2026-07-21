## Overview

本变更将 sequence collate 的参数来源从 dataloader config 隐式注入改为 collate block 显式声明。目标是让维护者只查看 `collate` / `train_collate` / `eval_collate` block 就能理解除 batch 外所有传入 collate function 的参数。

## Current State

此前 `SequenceDataModule._build_collate_fn()` 会执行：

```python
partial(
    curr_config.collate_fn,
    labels=curr_config.labels,
    sequence_length=curr_config.sequence_length,
    masking_token=curr_config.masking_token,
    padding_token=curr_config.padding_token,
)
```

这导致参数在 YAML 中分散在 dataloader block，而真正的 collate block 只显示 `_target_` 和少量函数专属参数。

## Design Decisions

### 1. datamodule 不再注入 collate 参数

统一后的 `BaseDataModule._build_collate_fn()` SHALL 直接返回 `curr_config.collate_fn`。这让 collate 参数绑定完全由 Hydra partial 负责。

### 2. 删除 dataloader config 中的 collate 专属字段

`SequenceDataloaderConfig` SHALL 删除：

- `labels`
- `sequence_length`
- `masking_token`
- `padding_token`

这些字段不是 dataloader 构造 `DataloaderWithIterationRetry` 所需字段，也不应再作为 datamodule 注入来源。

### 3. TIGER 配置将参数移动到 collate block

`tiger_train` 中：

- `train_collate` 声明 `labels`、`sequence_length`、`masking_token`、`padding_token`、`sequence_field_name`、`sid_hierarchy`、`max_batch_size`。
- `eval_collate` 声明 `labels`、`sequence_length`、`masking_token`、`padding_token`。
- train/val/test dataloader block 只引用对应 `collate_fn`，不再声明 collate 参数。

`tiger_inference` 中：

- `collate` 声明 `id_field_name`、`sequence_length`、`padding_token`。
- predict dataloader block 不再声明 `labels`、`masking_token`、`sequence_length`、`padding_token`。

### 4. collate function 签名保持稳定

`collate_fn_train`、`collate_with_sid_causal_duplicate`、`collate_fn_inference_for_sequence` 的函数签名已经显式表达除 batch 外的参数，通常无需改动。实现只改变参数绑定位置。

## Risks

- 若某个 sequence dataloader 配置未迁移 collate 参数，Hydra instantiate 可成功但 dataloader 运行时会因缺参失败；需要 grep 所有 sequence data YAML。
- 当前只有 TIGER 使用 sequence label/masking collate contract；仍需验证 `tiger_train`、`tiger_inference`。
- 删除 `SequenceDataloaderConfig` 字段是 breaking cleanup，但符合用户确认的目标。

## Non-Goals

- 不改变 collate 的 batch 数据结构、label 生成语义、mask 语义。
- 不改 item-level datamodule/config；item 链路已经直接返回 collate partial。
- 不重新设计 label function 体系。
