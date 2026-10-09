## Context
用户明确全模型随机初始化，并采用同一次训练内基础能力与高位目标适应方案。推荐模型随机初始化；SID/商品向量复用。50k更新总预算，25k基础联合训练；之后冻结本次teacher，128个training微批次标定cap，然后保持基础目标加capped NDCG及dual保护各weight1。无boundary/新梯度重分配。
## Decisions
单GPU固定协议，batch128累积2有效256；每微批随机取8个训练样本在线排序，候选为当前混合beam20与contentTop20并集并训练补正例，每次刷新。全部训练样本参与基础目标，不限旧3454。teacher不进optimizer、始终eval，切换后不更新，checkpoint存teacher及cap累计/阶段状态。cap校准仅使用冻结teacher的training候选及标签，128批后冻结，不消费validation。
训练小候选排名是原全候选loss的近似；推理保持beam20+全部cold，按混合概率终排，无补正例。同池content为辅助比较。验证重新读取原evaluation并按既有hash固定一半selection；audit另一半只在predict。50000更新包括预热，validation每2500更新（单卡累积2即5000微批）。只有完成calibration的checkpoint可作为最终best，前期selection monitor=-1。
## Risks
多项训练条件改变，无法唯一归因于训练轮数；teacher质量、短列表偏差、共享参数冲突、全模型训练内存与延迟均需保留。生产小批零optimizer探针验证两阶段和梯度，不代替完整训练效果。先单GPU支持，拒绝意外多卡避免阶段/评估状态不一致。
## Validation
CPU最小真实模型测试随机初始化/基础等价/全模块梯度/teacher冻结/校准不被eval影响/恢复和篡改拒绝；短候选训练注入及推理不注入；原split一致；Hydra与脚本边界；node1零更新真实batch与同步hash。

## 双卡扩展（2026-10-03，替代上述单卡限制）
用户明确使用 GPU 2、3。独立 experiment `copmrec_scratch_train_ddp2` 使用 DDP，每卡batch64、累积2，每卡排序4例；有效batch256、每更新排序16例、128个校准微批（全局1024例）、64个校准更新及2500更新验证间隔保持。每个校准微批all-reduce增量sum/count，各rank保存相同状态；teacher从已同步学生冻结，复制发生在学生前向之前。禁用DDP buffer广播，避免覆盖校准状态及不等长验证分片引入逐批collective。验证/预测按rank步进分片不补齐，验证结束all-reduce全局累计统计。保留单卡配置兼容；双卡checkpoint单卡audit须指定ranking_batch_size=4匹配契约。相同预算不表示两种并行执行逐位等价。
