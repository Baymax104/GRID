## Why

MIR validation 在推荐结果完成后仍逐前沿节点计算并丢弃概率证书，CPU 目录规模复现一批 16 用户产生约 3.28 万次成员查询。用户已停止并删除首轮运行，需要先解决实现瓶颈再重启完整比较。

## What Changes

- validation 和无需 trace 的标准推断只执行生成推荐所需计算。
- 正式 trace 使用静态 item 祖先索引批量计算概率上界，减少逐节点设备同步。
- 保持训练目标、Q/S、搜索调度、Top-K、数据 split、验证频率及启动命令不变。
- 增加预测等价、概率界等价、目录规模复杂度及 CPU 性能验收。

## Capabilities

### New Capabilities

- `item-resolution-efficient-evaluation`: 无冗余证据计算的验证路径和批量概率证书。

### Modified Capabilities

无。已有 MIR 提案尚未归档，本次独立约束其性能实现。

## Impact

限于 `src/recommendation/tiger_item_resolution/`、对应测试及实验说明；不修改 TIGER baseline 或引入依赖。不自动启动 GPU 实验、不操作用户已删除的 W&B run。
