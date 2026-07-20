## Why

当前 data 新架构已经把 dataset 的主链路收敛为 row-based：reader 通过 `iterrows()` 逐条产出样本，dataset 逐条执行 preprocessing，batch 组装职责则下放到 `collate_*`。

但 `src/data/components/preprocessing.py` 中仍残留旧的 batch 兼容语义：

- 函数参数与注释仍大量使用 `batch_or_row` 一类命名；
- 个别函数仍显式兼容 `list[dict]` 形式的 batch rows；
- 文档与实现边界容易让维护者误以为 preprocessing 仍需同时承担 row 与 batch 两种职责。

这与当前目标架构不一致。需要将 preprocessing 整体收敛为 row-only API，明确：preprocessing 只处理单条 row，batch 语义只存在于 collate 层。

## What Changes

- 将 `src/data/components/preprocessing.py` 中的 preprocessing 函数整体收敛为 row-only 接口
- 删除对 `list[dict]` / batch rows 的旧兼容分支
- 保留对 row 内字段值为 `list` / `np.ndarray` / `torch.Tensor` 的必要处理
- 同步更新参数命名、docstring 与注释，移除 batch / row 混合语义
- 保持 collate 层继续负责 batch 组装，不将该职责回流到 preprocessing

## Capabilities

### New Capabilities
- `row-only-preprocessing-contract`: 规定 preprocessing 仅处理单条 row，不再兼容 batch rows

### Modified Capabilities
- `dataset-owned-preprocessing-assembly`: 补充 preprocessing 运行时契约为 row-only，进一步清晰 dataset / preprocessing / collate 的职责边界

## Impact

- 受影响代码：`src/data/components/preprocessing.py`，必要时少量注释/文档同步修改
- 受影响范围：所有复用该 preprocessing 模块的实验链路
- 不改变 collate 输出结构，不改变 dataset 的 row-based 主链路，只去除旧 batch 兼容语义
