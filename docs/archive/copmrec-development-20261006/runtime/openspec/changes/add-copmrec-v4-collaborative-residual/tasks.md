## 1. 实现

- [x] 1.1 添加不改变v0默认数值的共享内容残差hook和v4子类。
- [x] 1.2 添加严格v0 weights-only初始化及v4恢复/来源记录。
- [x] 1.3 添加model/train/inference配置、薄入口脚本与scale0匹配对照。

## 2. 验证与运行

- [x] 2.1 CPU等价、非零梯度、cold mask、checkpoint拒绝及旧版本回归。
- [x] 2.2 Hydra compose、shell quoting/override透传和OpenSpec strict。
- [x] 2.3 Mutagen同步、remote哈希和真实双卡dry-run。
- [x] 2.4 记录授权、基线、累计预算，启动首个双卡续训并确认存活句柄。

## 3. 效果评估

- [x] 3.1 验证best与匹配续训对照，单卡Testing独立复算及增益/损益报告（首个dense Testing +5.26%/+5.64%，未达原10%目标；预承诺最后训练槽单独记录）。
