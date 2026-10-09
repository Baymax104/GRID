## Why

建立以 LIGER 为主要 baseline 的正式比较，需要可核查的官方方法适配，不能沿用 CGBS hybrid 冒充。

## What Changes

新增独立 Liger Lightning 模块，复用 keyed SID/content loader 与现有 SID batch；保留官方共享残差 MLP、位置编码、最后有效 token query、全目录 cosine CE 与 T5 SID CE。

## Capabilities

### New Capabilities

- `liger-content-model`：LIGER 内容增强与联合训练。

### Modified Capabilities

无。

## Impact

src/recommendation/liger/、src/data/components/liger.py、tests/recommendation/test_liger.py。保留现有用户改动，不启动完整实验。
