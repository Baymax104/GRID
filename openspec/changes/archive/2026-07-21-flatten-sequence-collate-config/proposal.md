## Why

Sequence dataloader 此前通过 `SequenceDataModule._build_collate_fn()` 隐式把 `labels`、`sequence_length`、`masking_token`、`padding_token` 注入 collate function。维护者查看 data YAML 时无法只从 `collate` block 判断 collate 的完整参数来源，容易造成配置参数双来源和误解。

## What Changes

- 将 sequence collate 的非 batch 参数全部展平到 collate 配置 block 中。
- 统一后的 `BaseDataModule` 不再隐式 partial 注入 `labels`、`sequence_length`、`masking_token`、`padding_token`。
- `SequenceDataloaderConfig` 删除不再由 datamodule 消费的 collate 专属字段。
- TIGER train / validation / test / inference 配置迁移为 stage-local collate 参数声明。
- **BREAKING**: sequence dataloader config 不再接受 `labels`、`sequence_length`、`masking_token`、`padding_token` 作为 datamodule 注入字段；这些参数必须由 `collate_fn` 的 Hydra partial 配置显式提供。

## Capabilities

### New Capabilities
- `sequence-collate-config-locality`: sequence collate 参数必须在 collate 配置处显式声明，datamodule 不再隐藏注入。

### Modified Capabilities
- `tiger-sequence-data-contract`: TIGER sequence 实验的 collate 参数来源改为 collate block 本地声明。

## Impact

- Affected code:
  - `src/data/datamodules/sequence.py`
  - `src/data/components/config_models.py`
- Affected configs:
  - `configs/data/tiger_train.yaml`
  - `configs/data/tiger_inference.yaml`
- Validation focus:
  - TIGER train/inference Hydra compose + instantiate。
  - `BaseDataModule._build_collate_fn()` 返回已配置好的 collate partial，不再二次绑定参数。
  - `SequenceDataloaderConfig` 不再暴露 collate 专属字段。
  - collate smoke 覆盖 train/eval/inference 参数仍按预期传入。
