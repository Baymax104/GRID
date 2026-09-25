# 冻结协议

- C来源ld5ecf58/39000，SHA c1fcc198e5391bd16c1a3bb3a66024e076b27366e2241f4e70a361c8a1e3d50e，原temperature0.1、PCA128、4原型及模型state。
- 已有A users.json SHA 768c0b3e1c2acc7ecb4bb98c4cbeed321cca6181bb898ce2c368cb5fd3594cbc，读取在data层，逐key/input_sha256/target_sid严格匹配，原off/zero有效才允许复用。
- evaluation512，seed20260923，单FP32，batch1、无梯度、无beam、无训练。dry-run2用户，不发布证据。
- 每用户一次encoder/query；全目录12101item scores=q·e/temperature；精确root为同一scores按SID首token做logsumexp。prototype调用原生产catalog.log_mass，不替换为归一化cosine。
- 全目录/目标root内的rank、NLL、Top10描述；root精确/原型完整scores保存。rank采用目录排序稳定打破同分并报告rank上下界，Top10采用同一稳定规则。
- 对精确与原型root NLL差作2000次用户bootstrap(seed20260924)。平均改善及95%下界>0，且精确rootTop10不下降，才eligible_for_aggregation_review；否则stop_root_aggregation_hypothesis。任何通过不等于超过A或许可新增训练。
- 输入标签只用于打分完成后的指标，不作为query或候选选择输入。条件item排名是目标root内的描述性oracle条件，不是部署结果。
- 输出完整用户统计、root分数、item分数SHA和top10item keys、来源与冻结state SHA；不保存完整512x12101矩阵。
- 保持先前失败结论。预计6195712个128维item内积和512次encoder/query前向；无新增训练/完整beam。
