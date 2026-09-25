## Why

建立以 LIGER 为主要 baseline 的正式比较，需要可核查的官方方法适配，不能沿用 CGBS hybrid 冒充。

## What Changes

依赖 add-liger-content-model，实现官方全词表生成、合法 SID 解析、冷启动商品补入及纯 cosine 重排；安全拒绝非法 SID 映射为商品0。

## Capabilities

### New Capabilities

- `liger-hybrid-retrieval`：LIGER 生成后内容检索。

### Modified Capabilities

无。

## Impact

src/recommendation/liger/、tests/recommendation/test_liger_retrieval.py。保留现有用户改动，不启动完整实验。
