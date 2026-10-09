# 冻结协议

基于2026-09-22-cgbs-post-objective-route-review.md，仅实施新信息资格诊断。A来源、SID、embedding、50k缓存不变，单FP32进程。

- training用户按SHA256("20260924:user:<id>")排序取2048；每用户按仅input_ids/mask的哈希选择一个窗口，同哈希按cache key排序。选择不看label或A是否失败。频次为完整50k training缓存目标首token计数，明确是窗口加权频次，不是完整训练集用户频次。
- evaluation沿用512用户、seed20260923；只做root，不做完整beam。全部256合法root均参与CE及评价。
- 控制特征：log1p(缓存目标频次)、log(目录分支商品数)，减候选均值；不标准化。内容4维为全部/最近4个历史item内容均值与prefix prototype的cosine mean/max。先归一化两个历史均值及prototype，零均值按零向量处理。
- 真实bank为checkpoint固定PCA128向量；置乱seed20260924，同时置乱历史item映射和目录item映射，重聚类仅root，最多4原型/5次迭代。A目录保持不变。
- 三臂offset均为原A root log概率、系数固定1，无bias；控制2参数，真实/置乱各6参数；零初始化、Adam lr0.01、200全批CPU更新、L2=0.001*参数平方均值，不选点。
- 拟合必须在torch.inference_mode(False)+enable_grad上下文创建CPU普通tensor，兼容Trainer.test默认inference_mode；仅线性系数参与优化。
- 在evaluation标签可用前拟合；评价不更新权重。保存训练user/key/窗口SHA、频次、permutation SHA、拟合轨迹、最终系数及逐用户完整候选分数。
- 用户bootstrap2000次，seed20260924。真实NLL分别优于控制/置乱且两项95%差值下界>0；root Top10存活不低于A；来源/完整性成立。否则停止，不扩预算。
- dry-run仅16训练用户/2评价用户/1次拟合更新，无bootstrap证据发布；正式要求固定预算与唯一用户数。

# 风险和边界

只检验这套固定内容特征的线性条件增量，不证明CGBS整体无用或搜索独特性。评价属于开发数据。不得以只选A失败gold与cutoff的二分类制造标签泄漏。新增1诊断作业内含3次小型拟合，不称零训练。
