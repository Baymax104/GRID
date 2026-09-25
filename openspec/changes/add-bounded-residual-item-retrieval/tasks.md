## 1. 模型与训练

- [x] 1.1 核对旧 dense 契约并完成设计与规格
- [x] 1.2 实现独立目录、基础模型、残差模型及共同候选目标
- [x] 1.3 实现基础初始化、恢复与来源校验

## 2. 标定与检索

- [x] 2.1 实现训练历史抽样、标定产物及读取
- [x] 2.2 实现reference/fixed/dynamic检索和边界证据

## 3. 运行与验收

- [x] 3.1 配置四种arm与训练/标定/推理/audit手动入口
- [x] 3.2 内存行为测试、Hydra配置、shell解析及OpenSpec strict验证
- [x] 3.3 更新执行说明与研究状态，flush同步并交付手动命令

验收：60项聚焦测试通过；Ruff、Git diff whitespace与OpenSpec strict通过。Mutagen flush成功，四个session均Watching for changes，无冲突。GPU实验未启动，未提交或归档。
