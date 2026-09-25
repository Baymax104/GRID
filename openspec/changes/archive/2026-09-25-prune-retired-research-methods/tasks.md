## 1. 冻结活跃方法接口

- [x] 1.1 将固定 log-probability mixture 收拢到 LIGER 核心并保持 CoPMRec checkpoint 参数兼容
- [x] 1.2 从 LIGER/CoPMRec 删除动态 gate、学习缓存、偏好分散、深度条件和外部 gate 入口
- [x] 1.3 更新活跃 LIGER 配置并验证主方法、baseline 和机制控制装配

## 2. 删除已结题方法垂直切片

- [x] 2.1 删除已结题 LIGER 变体的实现、数据模块、writer、配置、脚本和测试
- [x] 2.2 删除 BRIR 的实现、配置、脚本和测试
- [x] 2.3 删除 MIR/item-resolution 的实现、配置、脚本和测试
- [x] 2.4 删除 CGBS/catalog-grounded 及 training probe 的实现、配置、脚本和测试
- [x] 2.5 清理 Artifact loader、共享模块和配置中的悬空引用

## 3. 结构与回归验证

- [x] 3.1 增加活跃方法表面结构测试，禁止淘汰入口回流
- [x] 3.2 运行 LIGER、TIGER、Artifact、launcher 聚焦测试与 Hydra compose
- [x] 3.3 运行全量 pytest、Ruff 和 shell 语法检查
- [x] 3.4 运行 `openspec validate prune-retired-research-methods --strict`
- [x] 3.5 检查 Git diff，确认未覆盖范围外的用户修改
