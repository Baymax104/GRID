## Why

真实 Tokenizer batch1024 的四层 Diversity 采样产生4136次 CUDA 标量转整数，造成频繁同步。诊断中的批量原型保持输出/梯度逐位相同，前向加反向中位从117.8 ms降至22.4 ms，需将已验证的最小修复应用到生产实现。

## What Changes

- 批量读取 labels/ids，在 CPU 上按原索引顺序组织 group 成员并采样。
- 保留 Python random.choice 的调用、显式 positives 接口、错误检查和 loss 计算。
- 添加采样随机状态、梯度、短 optimizer 轨迹和读取次数回归；同步 node1 后有限核验真实 checkpoint。

## Capabilities

### New Capabilities

- `letter-batched-diversity-sampling`: Tokenizer 采样的批量读取及等价性要求。

### Modified Capabilities

无。

## Impact

仅 LETTER tokenizer、相应测试与证据文档。无新依赖、state_dict 或 checkpoint identity 变化；保持20000 epoch、分组频率、n_jobs10和Sinkhorn50轮。不启动或重启正式实验。
