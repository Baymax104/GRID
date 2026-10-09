## 1. 模型与概率

- [x] 1.1 实现 keyed 目录索引、内容特征和 checkpoint 指纹
- [x] 1.2 实现因果 prefix state、解析门、精确边缘目标与固定深度消融
- [x] 1.3 实现预算概率累积推断、下界与剩余质量
- [x] 1.4 实现 mask/init/dense/hybrid/COBRA 适配控制与 WIDE 标定/推断

## 2. 产物与配置

- [x] 2.1 实现独立 resolution trace schema 与兼容的共享 writer 校验扩展
- [x] 2.2 完成 train/inference/calibration 组件配置及 best/last、lineage
- [x] 2.3 完成根脚本、两组矩阵队列、单卡评价与 diagnosis 指令

## 3. 验证与交付

- [x] 3.1 验证概率、梯度、因果性、身份、预算和产物回归
- [x] 3.2 验证全部 arm 的 Hydra 装配、脚本语法与参数透传
- [x] 3.3 完成实验协议、研究状态同步与 OpenSpec strict 验证
