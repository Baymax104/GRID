## Why

v5随机初始化连续50k的完整Validation已审计：同预算native42下R10+3.5978%（CI跨零），N10+11.2897%（CI正），Top5双指标CI正，原双8门槛false。保留v5，符合用户允许明显部分收益作为整体改进基础的要求。919新增／841丢失／净78及固定排名桶显示前排收益伴随命中交换；旧64k双独立模型pool覆盖更高，只是互补表示的正面线索，不能证明内容侧或残差导致失例。

当前具体问题是：一个连续50k模型能否在保留前排收益的同时减少Top10命中丢失。仅检验一个固定的同query双目录评分规则，不扫描alpha、残差尺度、权重、checkpoint或seed。

## What Changes

- 新增v5.1薄模型：同一history query；目录投影一次，分别归一化内容目录和内容加seen残差目录，固定0.5/0.5平均其向量后一次matmul，平均向量不再归一化。
- final logits同时用于content CE、mixture NLL和dense部署；SID CE、learned alpha、history residual、cold0、raw checkpoint选择及history资格不变。
- 新增薄model/experiment配置及根脚本，训练随机初始化单次50k，原v5 runtime文件保持原字节。
- 复用严格来源／full-state／预算契约，v5.1不得恢复v5或warm权重。
- 预算用途显式修订：原3run/150k账本内，已用v5的一次50k；从余下2次中仅指定一次给v5.1 seed42/50k及一次完整单卡Val。余下1次用途未分配；原native43/candidate43配对无法在这次迭代后靠单个槽完成，不宣称双seed闭环完成。新增Test0，不重置任何旧额度。

## Impact

只新增模型、配置、根脚本和聚焦测试，并扩展本地证据编排以支持v5.1版本/source/checkpoint。零新增训练参数、无第二次T5前向；增加向量归一化和平均操作，不声称FLOPs或耗时相同。旧v5、CP、输出、gate及原负向/不确定证据均保留。
