## Why

候选并集在selection上相对mass20新增49、损失40个Top10命中，新增目标全部是max-only；union Top10平均含1.84个max-only候选。统一mass来源bonus没有形成稳定平台，需要区分“来源信号无用”与“连续加分不能直接约束候选竞争”。

## What Changes

- 复用现有evaluation缓存，固定检验Top10最多一个max-only增量候选。
- mass20与cold构成主体；max20中不属于mass20且不属于cold的候选最多保留内容分数最高的一个。
- 使用既有selection/audit划分；selection失败时不计算audit比较。
- 不扫描quota、阈值、分数权重、用户子群或testing。

## Impact

只增加缓存分析函数、聚焦测试、冻结协议和结果证据。成本为0训练、0 prediction、0 testing。
