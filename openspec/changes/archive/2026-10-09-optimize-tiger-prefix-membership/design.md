## Context

现有实现对每组候选和完整 catalog 做三维广播比较，工作量和临时存储随两者乘积增长。优化必须保持每个前缀的布尔判定，从而保持原有解码逻辑。

## Goals / Non-Goals

**Goals:** 降低合法 SID 查询开销；保证精确查询、缓存一致性和旧 checkpoint 兼容。

**Non-Goals:** 修改 beam search、softmax、top-k、encoder 调用、KV cache、验证频率或数据协议。

## Decisions

1. 使用 radix=codebook_size 的 int64 编码。对深度 d，仅在 K**d - 1 不超过 int64 上限、catalog token 全部位于 [0,K) 时启用。该范围内编码一一对应；其他情况退回逐 token 比较。
2. 逐深度惰性构造排序唯一键，searchsorted 后同时检查位置和键相等。候选先检查范围，防止非法 token 通过编码碰撞命中；浮点等非常规候选类型走原始比较。
3. 缓存为普通 Python 属性，不进入 state_dict。记录 catalog 对象、版本计数和 codebook_size；设备迁移触发 catalog 替换并失效。无版本计数的 inference tensor 每次重建，以避免遗漏原地修改。
4. 保留原始比较作为兼容路径和独立测试 oracle，不修改任何候选排列和分数计算。

## Risks / Trade-offs

- 首次索引构建有开销 → 后续查询复用，真实探针区分构建与稳态耗时。
- 非常规 catalog 仍使用高开销路径 → 保持原行为，明确测试 overflow 和非法 token。
- 并发线程修改 catalog 或使用 .data 绕过版本计数不属于支持的更新方式 → 使用正常 tensor 原地操作或替换属性；正式运行 catalog 固定。
- 有限 batch 不能替代完整评价 → 用数学边界、差分单测和真实多 batch 证据限定结论。

## Migration Plan

旧 checkpoint 直接 strict 加载，无格式迁移。通过本地聚焦测试和远端有限组件探针验收；完整实验由用户手动启动。
