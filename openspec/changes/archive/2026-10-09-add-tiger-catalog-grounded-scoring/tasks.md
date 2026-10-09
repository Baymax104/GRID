## 1. 规格与固定目录

- [x] 1.1 核对 baseline、统一入口和实验契约，完成 proposal/design/spec。
- [x] 1.2 实现 keyed 内容加载、确定性 PCA、多原型前缀索引与来源身份。

## 2. 独立方法模块

- [x] 2.1 实现八条件、合法分支训练、辅助 CE 和有界混合。
- [x] 2.2 实现合法唯一生成、兼容 prefix trace、Hybrid 推荐及 checkpoint 校验。

## 3. 实验装配

- [x] 3.1 新增训练/推理 component 与薄 experiment 配置，保留 baseline 默认行为。
- [x] 3.2 新增训练、推理和两组队列脚本，验证参数、notes、seed、dry-run 和 override。
- [x] 3.3 完成研究实施文档、条件矩阵、判定规则和手动启动命令。

## 4. 验证

- [x] 4.1 通过固定索引、概率、梯度、生成、trace 和 checkpoint 聚焦测试。
- [x] 4.2 通过所有条件 Hydra compose、脚本测试、相关 baseline 回归和静态检查。
- [x] 4.3 通过 OpenSpec strict 校验并报告未执行的 GPU 实验边界。
