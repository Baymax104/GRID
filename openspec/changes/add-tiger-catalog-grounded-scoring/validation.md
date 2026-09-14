# 实施验收记录

日期：2026-09-12。10/10 实施任务完成；完整实验未启动，方法效果未验证。

## 聚焦验证

```bash
uv run --no-sync pytest tests/recommendation/test_tiger_catalog_grounded.py tests/test_tiger_catalog_grounded_config_script.py -q --tb=short
```

最终结果：**67 passed**。覆盖八条件训练/推理的 Hydra 实例化、真实微型 T5 反传、全部参数梯度、原模型一致性、CPU/CUDA RNG 隔离、质量近似界、多原型质量守恒、合法唯一 beam、teacher/beam 概率、现有 prefix trace schema、checkpoint 校验、key 对齐、显式 diagnosis split 和四个脚本。

## 相关回归

```bash
uv run --no-sync pytest tests/recommendation/test_tiger_catalog_grounded.py tests/test_tiger_catalog_grounded_config_script.py tests/recommendation/test_tiger_metrics_runtime.py tests/recommendation/test_tiger_prefix_trace.py tests/data/components/test_artifacts.py tests/test_launch_script_data_seed.py -q --tb=short
```

联合结果：**160 passed**，该次包含当时的 58 项新增检查和 102 项既有相关检查。随后补充多原型数值界与八条件 inference 实例化，最终新增检查为上面的 67 项；两次覆盖合计 **169 项不同测试**，不是 227 项。只有现有依赖弃用警告。

## 静态及规格验证

- 新增 Python 文件 Ruff check 通过，已运行 Ruff format。
- 四个 shell 脚本 Bash 语法检查、notes quoting、dry-run、必填参数、错误输入和 override 顺序测试通过。
- `openspec validate add-tiger-catalog-grounded-scoring --strict` 通过。
- `git diff --check` 通过；GRID 已跟踪文件没有修改，本变更为新增文件；既有 `tmp/` 保留。
- 研究 `research-state.yaml` 解析通过；实施文档与当前方案/执行看板同步。

## 实施中修正的设计口径

- 原 TIGER 生成已经 mask-before-softmax；mask_ce 隔离训练归一化。
- 新方法 CE 按层平均；辅助系数 0.1 对应该尺度。
- Hybrid 使用两路概率混合排序，避免退化为纯 dense Top-B。
- 新增独立 diagnosis 标签划分适配，避免无 trace 的 evaluation 推荐被默认按 testing 分析。
- 额外 query 初始化只修改被隔离的 CPU RNG，不重置 CUDA RNG。
- 训练/推理 task_name 包含条件与数据集，避免并发输出目录冲突。

## 输入核对与未验证范围

W&B API 只读确认 Beauty `i0py1978` 和 Sports `xvjro8g1` 的 data/SID/embedding 引用，完整命令见 `E:/projects/research/docs/cgbs-implementation-and-experiments-2026-09-12.md`。

没有运行完整 experiment、GPU/DDP 训练或真实目录性能测试；没有发布新实验、提交或推送代码。真实显存、吞吐、收敛、Head/Tail 效果和论文贡献仍由用户启动的首轮矩阵检验。该变更保留为已实施但尚未归档的 OpenSpec。
