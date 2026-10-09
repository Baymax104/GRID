# 设计
复用一次encode、dense logits和generation。在同一检索循环记录目标全目录/候选内稳定排序排名、生成候选row（-1无效）、候选数量、cold状态与TopK。完整候选集可由generated rows和固定seen_mask重构；不导出全目录分数。rank=0表示目标不在候选集。辅助产物携带用户key、标签SID和catalog指纹，由共享AuxiliaryTensorWriter合并发布。仅hybrid且有标签允许启用。不改变state_dict，不启动完整实验。
