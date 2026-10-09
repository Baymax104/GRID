## Context

Sinkhorn求解软平衡分配，逐行argmax不能保证hard code唯一；相同latent会得到相同分配。作者generate_indices.py允许20轮之后仍有冲突，不满足GRID原始商品唯一候选契约。

## Goals / Non-Goals

目标是在末层256容量内生成唯一四码SID，保留前三层学习码及近邻无冲突组。非目标：改训练目标、扩充第五位、声称完全相同于作者输出。

## Decisions

- 仅当作者式修复仍失败时启用fallback。按前三层prefix收集全部商品，不能只看当前碰撞商品而遗漏已占用末码的邻居。
- 对存在碰撞的prefix全组，计算latent减前三层码本和后的残差至256个末码的平方距离；SciPy linear_sum_assignment求矩形一对一最小总距离。严格按已排序商品key顺序作为输入，在相同环境重复导出确定性。
- 若prefix商品数大于256，明确失败；不偷偷改前层语义或增加位数。返回前检查全目录唯一性。
- SciPy已为k-means-constrained间接依赖，此处列为直接依赖；不升级已锁定版本。

## Risks / Trade-offs

末层硬匹配不同于作者软分配argmax → issue和文档标明GRID适配，只主张共同协议比较。容量不足仍可能失败 → 显式诊断prefix人口，不增加隐性码空间或调参搜索。

## Migration Plan

checkpoint结构及训练保持兼容；旧checkpoint用新导出协议重新生成SID，推荐checkpoint绑定实际SID字节。修复前后源码指纹分别登记，不把后续snapshot追补为早期来源。
