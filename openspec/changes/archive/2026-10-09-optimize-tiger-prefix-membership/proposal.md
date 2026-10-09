## Why

TIGER validation 的 catalog 广播比较占实际单 batch 用时约 83%，并产生约 13 GiB 的峰值显存分配。已有真实 checkpoint 探针表明，精确前缀索引可显著降低开销，需要把原型落实为具有等价性边界和回归验证的实现。

## What Changes

- 用逐深度的排序唯一整数键和 searchsorted 查询合法 SID 前缀。
- 检查 token 范围和 int64 编码边界；无法安全编码时保留原始比较路径。
- 缓存随 catalog 替换、原地修改或设备变化失效，不改变 checkpoint 格式。
- 补充与旧实现的差分测试及真实 checkpoint 的有限多 batch 验证。
- 不修改 beam 排序、概率、loss、评价协议或训练配置。

## Capabilities

### New Capabilities
- `tiger-prefix-membership-index`: 精确且兼容旧 checkpoint 的前缀索引查询。

### Modified Capabilities

无。

## Impact

影响 TigerDecoder 的前缀合法性检查及复用它的推荐模型。新增测试和性能验证记录，无新增依赖，无完整训练或评价任务。
