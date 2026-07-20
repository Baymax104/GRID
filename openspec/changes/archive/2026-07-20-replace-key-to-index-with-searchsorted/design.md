## Context

`load_model_output` 加载 `.pt` 文件后调用 `_build_key_to_index` 构建 `dict[int, int]`（业务主键 → 行号），挂到 `bundle["key_to_index"]` 作为运行时缓存。`gather_predictions_by_keys` 用这个 dict 做 O(1) key→行号翻译，再用行号索引 `predictions` tensor。

当前问题：
- `_build_key_to_index` 是 Python 循环，大 N 时慢
- `bundle["key_to_index"]` 是注入到 dict 的运行时字段，mutate 了加载结果
- `gather_predictions_by_keys` 包含懒构建逻辑（`bundle.get("key_to_index")` + fallback），实际从不触发
- `keys` 和 `predictions` 是顺序对应的 tensor，可以用纯 tensor 操作完成 key→行号翻译

## Goals / Non-Goals

**Goals:**
- 用 `torch.searchsorted` 替代 dict lookup，消除 `_build_key_to_index` 和 `key_to_index` 运行时字段
- 保持函数签名和外部行为等价

**Non-Goals:**
- 不改变 `gather_predictions_by_keys` 的函数签名
- 不改变 `.pt` 文件的磁盘格式
- 不改变 `deduplicate_rows_in_tensor` 的行为

## Decisions

### 决策 1：加载时按 key 值排序

**选择**：`load_model_output` 加载后用 `argsort` 按 key 值排序 keys + predictions。

**理由**：`torch.searchsorted` 要求输入有序。排序是一次性 O(N log N) torch C++ 操作，比 Python dict 构建的 O(N) 循环在大 N 时更快。排序后 keys 有序，`searchsorted` 可直接使用。

**替代方案**：不排序，用布尔掩码 `(keys.unsqueeze(1) == lookup_keys.unsqueeze(0))` 查找。否决：O(N×M) 时间和内存，大 N 不可接受。

### 决策 2：排序安全性

**选择**：排序改变 keys/predictions 的行顺序，但所有消费者不受影响。

**验证**：
- `deduplicate_rows_in_tensor`：全量读取 `predictions` 做行去重，与行顺序无关
- `_validate_model_output`：检查 type/shape，与顺序无关
- `gather_predictions_by_keys`：按 key 值查找，与行顺序无关
- 无消费者直接按行号索引 `bundle["predictions"]`

### 决策 3：searchsorted 边界校验

**选择**：`searchsorted` 返回的索引需校验 `keys[indices] == lookup_keys`，不等则 raise KeyError。

**理由**：`torch.searchsorted` 对不存在的 key 仍返回插入位置，需显式校验匹配。

## Risks / Trade-offs

- **[排序改变行顺序]** → 已验证所有消费者不受影响。风险低。
- **[重复 key 时 searchsorted 行为]** → `_validate_model_output` 不检查重复 key。当前 `_build_key_to_index` 会 raise。排序后 `searchsorted` 对重复 key 返回首个匹配位置，不 raise。需在 `load_model_output` 中保留重复 key 检查。
