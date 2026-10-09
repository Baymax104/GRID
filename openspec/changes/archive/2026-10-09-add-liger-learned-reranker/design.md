## Context
用户批准263→64→1残差排序器，冻结LIGER，s=d+tanh(MLP)，最后线性层零初始化，幅度1固定。原固定checkpoint无训练组合阶段8次预测已结题。
## Goals / Non-Goals
实现可手动执行、可审计的有限学习验证。不训练backbone，不扫参数，不使用evaluation或testing标签拟合。
## Decisions
缓存通过现有LIGER predict与统一main。training和evaluation各1次缓存，每用户使用该split最后一个完整训练目标（不新增随机窗口），保留user_id。哈希用户ID的10%作为内部验证，剩余90%训练。训练上限50000用户、evaluation全量；超过上限失败，不静默截断。缓存分片streaming，避免全部特征驻留内存。
两臂共同保留cold：joint=dense20∪mixture20∪cold；control=cold加按dense顺序补足到joint实际大小。两臂feature=归一化q*v、abs(q-v)、dense分数、完整SID loglikelihood、dense第10名边界差、两路池内倒数排名、dense20和mixture20标记。所有商品完整SID teacher forcing。标签只计算loss目标，池外不注入。
训练共用两臂目标均覆盖且非内部验证用户；连续三分数统计仅在上述训练样本的两臂有效候选上估计。其余特征不标准化。内部验证包含所有留出用户，目标未入池仍为0分。固定AdamW lr1e-3 wd1e-4 batch128，3epoch无调参，以内部验证NDCG10选择最佳；两臂同初始化和shuffle seed42。
新增data manifest解析、IterableDataset按分片洗牌；共享writer目录保存标准keys/predictions分片，manifest统计覆盖、用户隔离、归一化统计、源checkpoint内容指纹、每片SHA。Artifact下载仅由artifacts.py入口resolve_reference。训练checkpoint保存arm、训练cache指纹、normalizer与source identity；推理核对cache源身份，输出标准Top10与逐用户rank证据。
root脚本提供cache/train/inference三种阶段，所有入口src.main。cache保留checkpoint_reference并计算checkpoint SHA证明相同模型；两个cache源身份必须一致。dry-run禁用writer避免发布不完整产物。
## Risks / Trade-offs
单个末端训练目标覆盖可能不足 → 缓存报告共同覆盖量，数量未达固定门槛1000训练/100内部验证共同覆盖则阻止训练；不静默注入标签或加窗口。训练数据被backbone见过 → 内部验证不是系统独立泛化证据。缓存2臂FP32特征体积显著 → 分片与文件字节成本报告；单用户上界约154KB，50000用户约7.7GB，实际需缓存后确认。源文件信息和manifest哈希阻止跨run混用。
新阶段计划2次backbone缓存、2次轻量训练、2次缓存评价；条件失败即止，0参数搜索。先运行training缓存并检查覆盖/成本，再开始其余阶段；不自动启动任何完整实验。
