## Why

Beauty真实12101商品目录的有限验证中，作者式末层Sinkhorn修复20轮后仍有碰撞，阻塞BMX-58。作者容许重复SID，但GRID原始商品级完整目录协议需要唯一SID以区分商品。

## What Changes

- 保留作者20轮修复；剩余碰撞按三层前缀分组，末层做最小总残差距离的一对一分配。
- 不改变训练loss或添加第五位；容量不足仍拒绝导出。
- 明确登记为GRID导出适配，并验证重复内容、邻居已占末码和真实目录。

## Capabilities

### New Capabilities

- `letter-unique-sid-export`: 在末层容量内唯一化LETTER四码SID。

### Modified Capabilities

无；原LETTER提案仍未归档。

## Impact

LETTER tokenizer导出、测试、pipeline协议和Linear交付说明；SciPy由现有间接依赖提升为直接依赖，版本不变。不修改其他模型。
