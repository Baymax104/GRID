## 1. 盘点与定位

- [x] 1.1 识别 `src/data/components/preprocessing.py` 中仍使用 batch / row 混合语义的函数
- [x] 1.2 区分“batch rows 兼容分支”与“row 内 sequence-like 字段值处理”

## 2. 实现 row-only 收敛

- [x] 2.1 删除 preprocessing 中对 `list[dict]` / batch rows 的旧兼容分支
- [x] 2.2 将相关函数参数命名统一为 row-only 语义
- [x] 2.3 保留并必要时澄清 row 内 `list` / `np.ndarray` / `torch.Tensor` 字段值的处理逻辑

## 3. 同步清理描述

- [x] 3.1 更新相关 docstring，去除 batch / row 双模式描述
- [x] 3.2 更新注释，明确 batch 语义属于 collate 层而非 preprocessing 层

## 4. 验证收尾

- [x] 4.1 grep 验证 `preprocessing.py` 中不再残留 `list[dict]` batch rows 兼容分支
- [x] 4.2 import / smoke check：preprocessing 模块仍可导入并在当前 row-based 主链路中正常工作
- [x] 4.3 总结该 row-only contract 对后续 data 模块收敛的约束
