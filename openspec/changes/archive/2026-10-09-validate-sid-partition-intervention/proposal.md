## Why

E1 run dci0bx3o 通过冻结代理门槛，控制后相关 0.500926 为边界通过；尚未证明量化有害或改组能改善 item 排序。用户要求继续 E2，需要把代理转为可否证的干预，并隔离改动规模、频次与几何扰动。

## What Changes

- 增加一次 CPU 准备实验，training 内构造行为引导与匹配扰动的完整 SID 置换，先检查可行性与代理泛化。
- 两臂使用完全相同的改动 item 集合；在四个不同首层组的四 item 块内采用不同配对交换，保持完整 tuple 集合、唯一性及全部前缀目录占用。
- 成功才输出三套 keyed bundle；失败保留证据且不提供可训练映射。
- 提供三臂相同 mask_ce 随机初始化、20k 更新、best-val NDCG 的训练入口和输入证据校验；所有真实任务由用户手动启动。
- 共享 structured writer 增加可选 keyed tensor bundle 序列化，兼容现有 JSON/CSV 输出。

## Capabilities

### New Capabilities
- `sid-partition-intervention`: 受控 SID 置换准备、门禁、证据与三臂训练装配。

### Modified Capabilities

无现有规范的破坏性变更。

## Impact

新增 quantization 纯算法、data 装配与验证、experiment/component 配置、根启动脚本及聚焦测试。共享 structured writer 向后兼容扩展。不改 E1 参数，不新增依赖，不恢复其他暂停路线。
