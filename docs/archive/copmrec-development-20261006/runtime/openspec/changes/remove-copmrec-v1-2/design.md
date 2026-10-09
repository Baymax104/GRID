# 设计

显式删除7个v1.2专属运行/测试文件；恢复原head构造、四路拼接评分、v1/v1.1 trace和测试矩阵。原新增的simplify-copmrec-relevance-head规格移入docs/archive历史区，避免成为当前可执行契约；历史证据JSON保持原样。当前方法为v1.1，仍采用已采纳的校准权重，不回退为等权。

验证包含已有模型/loss/恢复回归、Hydra compose、Bash参数与语法、无v1.2活动引用，以及Mutagen删除传播和v1.1运行文件哈希。无需正式训练，也不读取或更改既有实验重资产。
