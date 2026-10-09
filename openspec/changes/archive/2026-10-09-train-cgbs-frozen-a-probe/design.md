## Context

用户采纳先验证稳定增量的建议。完整实验仍由用户手动启动，首先只准备 seed42/2k 探针。

## Goals / Non-Goals

目标是隔离在线内容增量，严格复用已训练 A 的全部主干与目录。此轮不实现 prefix query、新损失或多 seed 矩阵。

## Decisions

- 新子类接收 Artifact helper 返回的原生 checkpoint envelope；通过原目录契约只允许 token_content_init/full_content A，无额外 interaction。
- 校验原 checkpoint 全部 state keys、shape、dtype、finite 与离散目录；原始数据 fingerprint 核验输入身份，浮点目录以 checkpoint 为准以避免重算 PCA 平台误差。只有 content_query 与 mixture_logits 是新参数。先合并再 strict load，不使用宽松加载掩盖缺失。
- Encoder、Decoder、共享 SID 全冻结；覆盖 train() 保持主干 eval。只允许附加参数进入新 Adam，不继承 A optimizer/global_step。
- 保持原 generation CE + 0.1 item content CE；仅 head 与每层 mixture 受训练。mean/128、原 alpha 初始值和温度不变。
- checkpoint 保存原 A 引用、文件 SHA256、原 step 和冻结协议；恢复必须匹配。标准 CGBS inference 可严格加载同结构权重，额外来源保存在训练 checkpoint。
- 默认 2k 更新、每500验证，选择验证 NDCG best；预算检查禁止超过2k。评估后才决定重复实验。

## Risks / Trade-offs

- 2k 无收益不能否定所有内容机制；只停止扩大当前探针。
- 固定 A40k best 实际来源 step39000，计算成本按已有40k训练预算加2k报告。
- CPU有限测试不能代替真实数据全量 C-off/A 对齐；完整三臂 evaluation 仍是正式有效性门槛。
- 使用已消费 evaluation 作开发筛选，不声称独立测试确认或论文新颖性。
