## 1. 实现

- [x] 1.1 实现合法分支特征、零初始化门控及版本化checkpoint身份
- [x] 1.2 实现base/fixed推理契约和训练alpha统计
- [x] 1.3 增加薄model配置和现有脚本E条件，保持默认及透传

## 2. 验证与交付

- [x] 2.1 验证数值边界、初始等价、梯度、beam/teacher一致与checkpoint恢复
- [x] 2.2 验证统计聚合、Hydra配置、shell语法/参数与OpenSpec strict
- [x] 2.3 写运行说明，完成Mutagen flush/session核验，交付单次双卡命令

## 3. 科学验证

- [x] 3.1 用户手动完成E训练后核验与A/C的同口径结果，再按判据决定常量校准及冻结对照（见 training-outcome.md：NDCG基本持平C且仍低于A，不进入正式常量校准；不代表机制已证实）
- [x] 3.2 用户授权并手动完成单次E推理后，与已有A/C逐用户配对，分解命中和排名贡献（见 inference-outcome.md：问题已回答，未晋级主方法，不追加实验）
