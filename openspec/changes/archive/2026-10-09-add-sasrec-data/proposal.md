## Why

SASRec backbone 需要商品级序列，现有 TIGER/LIGER 数据链将商品映射成 SID 并扩展子序列，不能直接复用。原始商品0必须与 padding0隔离，负采样必须遵循作者 sampler。

## What Changes

- 新增固定目录原始 key 与1..N连续 ID 双向映射。
- 新增官方左 padding、移位标签、全 training 行排除的均匀负采样，以及只预测最后商品的评估预处理。
- 新增 SASRec batch 和 collate；复用 reader/dataset/datamodule，不改变它们。
- 仅覆盖数据模块，依赖已完成 backbone；Lightning 和配置不在本提案。

## Capabilities

### New Capabilities

- `sasrec-data`: 商品目录、序列监督与批次的数据契约。

### Modified Capabilities

无。

## Impact

新增 data helper，扩展 data_models/collate，新增内存测试。目录通过共享 Artifact loader 读取 bundle keys，不消费 SID 或内容 predictions。
