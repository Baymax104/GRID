## Why

四组 Tail 搜索—排序重分析表明，Tail 标签的主要瓶颈是逐层 prefix 生存和候选不可达；单纯扩大 beam 没有产生任何 Tail Top10 新增。下一步需要一个有边界、无测试标签泄漏的 candidate allocation probe，直接检验改善低训练支持 prefix 的候选保留能否转化为 Tail recommendation 增益。

## What Changes

- 新增可选的 prefix-balanced beam allocation：仅在合法候选的模型分数 shortlist 内，为训练支持较低的 prefix 保留少量 beam 槽位，其余槽位保持按原始路径分数选择。
- 从 training split 和 keyed semantic ID bundle 构造与 item 对齐的训练频次输入，并在运行配置中记录来源、策略参数和关闭状态。
- 扩展 TIGER prefix trace，使 intervention run 同时记录原始模型路径分数、allocation 选择信息和策略身份；关闭 probe 时保持现有输出逐元素一致。
- 扩展 Tail-SID diagnosis，支持同 beam width 的 baseline/intervention 配对审计和 candidate reach、Top10 新增/丢失、prefix survival、Head/Tail recommendation 变化证据。
- 新增独立的 probe experiment 配置与根目录启动脚本，支持 `--dry-run`、必填 `--data-dir`、默认 `--seed=42`、notes 与尾部 Hydra override。
- 将完整 Beauty/Sports × RKMeans/RVQ 实验保留为人工启动步骤；自动验证只运行单元测试、Hydra compose、脚本参数检查和 OpenSpec strict validation。

## Capabilities

### New Capabilities

- `tiger-prefix-balanced-allocation`: 训练频次驱动的 prefix prior、shortlist 内保留槽位、配置约束和默认行为兼容性。
- `tail-candidate-allocation-probe-analysis`: baseline/intervention 配对审计、分组候选与推荐变化、成功/退出门槛证据。

### Modified Capabilities

- `self-contained-tiger-generation-model`: TIGER decoder 增加可选的 decoder-owned candidate allocation 决策，但默认 generation contract 保持兼容。

## Impact

- 影响 `src/recommendation/tiger/decoder.py`、`src/recommendation/tiger/tiger.py`、训练频次数据 helper、prefix trace payload、Tail-SID diagnosis evidence 与相应配置、脚本、测试。
- 不新增第三方依赖，不改变 checkpoint 参数形状，不要求重新训练模型。
- 完整推理和在线 W&B 发布必须由用户手动启动。
