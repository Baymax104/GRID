# A深层范数校准：两seed诊断尝试

## 固定条件

本轮候选deep_norm_calibrated，eta=0.5，norm_floor=1e-8，版本deep-norm-calibration-v1。仅第二、第三语义层改变初始化范数，方向不变，再匹配原随机表population std。保留首层、去重层、未用码、参数量、损失与随机流；全层均值和欧氏距离并非保持量。近零范数截断，零向量保持零。

Beauty，seed45和46各一次，20k steps，每500steps验证，lr=0.0005，每卡batch128，两个进程。训练从头开始，复用原SID与内容产物，不运行额外A或残差基准。seed45参考A=kckgwr6j，seed46参考A=v7q7y0jx；配置自动记录同seed来源、物理GPU和协议。两个诊断seed为事后针对性选择，不能作为无偏跨seed收益估计。

## 手动启动

在node1仓库根目录运行，默认在GPU0/1依次训练两项；第一项失败则停止。

```bash
cd /data3/weizhenyu/projects/GRID
for seed in 45 46; do
  bash ./tiger_content_initialization_train.sh \
    --data-dir data/beauty \
    --condition deep_norm_calibrated \
    --seed "$seed" \
    --gpus 0,1 \
    --master-port 29760 \
    --notes "A deep norm calibration; eta=0.5; paired existing A; diagnostic seeds45-46; no sweep" || break
done
```

固定指数来自model组件，无需重复Hydra override。脚本保留用户override最后优先的契约，改变参数会偏离本计划，结果分析时必须重新核对resolved config。推理组件tiger_norm_calibrated_initialization_inference已验证兼容，但本轮不启动推理。

## 判读与停止

主指标best-val NDCG@10；同时检查20000步、后五个验证点均值及Recall。两个seed均超过各自A，且后段也改善而非只有孤立峰值，才考虑补齐其他三个seed。混合或仅微小峰值收益不支持机制成立，不自动扫eta或添加模块。先检查完成状态、预算和上游身份；版本确定前不打开testing。

## 验证记录

- 初始化/加载/数值边界测试27项通过；脚本与配置聚焦测试35项通过，含bash语法、参数quoting、空值与错误输入、dry-run透传、override优先级、两卡映射、同seed参考身份。
- 原catalog模型回归86项通过。合计148项，仅内存模型和Hydra compose，不启动完整实验或W&B run。
- eta=1权重逐位复现原A；eta=0.5深层方向、幂律范数比与全层std匹配通过，其他层/参数/随机流保持，梯度有限。checkpoint拒绝错误模式和指数。
- 历史frozen-implementation.json保留；本轮另存norm-calibration-implementation.json，不将更新后的模块文件hash伪称历史未变化。
- OpenSpec严格校验通过；Mutagen flush成功，随后四个session均Watching for changes，无conflict。仅交付代码，完整训练由用户手动启动。
