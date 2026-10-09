## Context

v1每个排序用户约70个候选，64分块导致8次候选decoder前向；排序增加的反向成本同样显著。当前训练仍在运行，已有v1必须能继续复现。

## Goals / Non-Goals

目标：新增v1.1跨用户批量评分，训练和推理统一使用；减少decoder调用，并验证有限显存、评分、梯度、checkpoint与DDP契约。

范围外：变更候选集合、排序抽样、beam、loss、评价用户与频率；冻结组件、持久化旧模型候选、完整训练。cross-attention K/V复用需修改T5内部实现，本次先实现直接可验证的批量化，不引入自定义attention。

## Decisions

1. 在v1抽出候选列表评分边界，默认实现仍逐用户；v1.1子类重写批量实现，其他主干、候选、训练/trace复用。基于相同权重比较v1行为。
2. 将每用户候选列表拼接为`(owner,row)`，按全局最多256对分块，使用`encoded[owner]`保留autograd，再按原列表长度拆回。与增加单用户chunk相比，此方式也合并不足一块的不同用户，支持尾批和不等长列表。
3. v1.1具有独立版本和执行契约，checkpoint记录全局chunk大小，禁止当作v1直接恢复。可继续使用既有v0 weights-only初始化，所有参数持续训练。
4. 使用独立`copmrec_v1_1_{train,inference}`入口。完整推荐实验仍由用户手动运行。

## Risks / Trade-offs

- 批量化改变dropout随机数分配和浮点归约顺序 → eval/dropout0做数值与全参数梯度对照，训练dropout非零做有限梯度和更新检查，不声明轨迹逐位一致。
- 全局256分块增加同时活跃的候选张量 → 保留可配置chunk，实测峰值allocated；完整DDP optimizer峰值单独保留未验证说明。
- 动态候选数量与rank数据不同 → CPU两进程DDP连续更新核验，保持全部参数参与同一步图和全局指标同步。
- 共享GPU扰动计时 → 同批、同模型权重预热后交替测量，并明确零更新与资源负载边界。

## Migration Plan

保留v1入口和checkpoint；新训练选择v1.1。Mutagen按既有三会话边界同步，flush/status及运行文件哈希确认，不处理远端.git或依赖安装。失败时继续使用v1入口。
