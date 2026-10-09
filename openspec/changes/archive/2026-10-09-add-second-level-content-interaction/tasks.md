## 1. 固定目录组件

- [x] 1.1 实现物品等权加性拟合、连通分量gauge、收敛检查和hash记录；用不均衡/不连通内存目录验证正交分解。
- [x] 1.2 实现interaction/additive/shuffled固定buffer及RMS校准，验证局部随机流、重投影及退化检查。

## 2. 模型集成

- [x] 2.1 添加只含一个共享标量的可选二层输入变换，保持原A参数、默认路径和全局随机流。
- [x] 2.2 覆盖encoder、teacher forcing和CGBS实际beam三条路径；验证padding/SEP、因果性、局部评分一致性。
- [x] 2.3 实现新增checkpoint契约与推理buffer恢复，检查optimizer/DDP单次注册及历史A兼容。

## 3. 配置与操作契约

- [x] 3.1 增加薄model配置和显式单条件根脚本，保留src.main、data-dir、seed、notes、dry-run及override协议。
- [x] 3.2 按experiment-plan核验可复用A身份，生成真实可执行手动命令与有限预算说明，不启动实验。

## 4. 验证与交付

- [x] 4.1 运行聚焦uv run pytest、Hydra compose、shell语法与参数测试、Ruff及strict OpenSpec校验，记录结果。
- [x] 4.2 更新实施状态与风险，保持完整训练、推理、远端同步和多seed扩展为独立明确操作。
