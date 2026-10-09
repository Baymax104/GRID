# MIR 实施验收

日期：2026-09-13。当前结论：提案及实现完成，完整实验等待用户手动启动；没有方法有效性或真实 GPU 成本结论。

## 聚焦验证

最终四组测试共 **93 passed**，4 条第三方依赖弃用提示，无失败：

```powershell
uv --cache-dir tmp/uv-cache run --no-sync pytest tests/recommendation/test_tiger_item_resolution.py tests/test_tiger_item_resolution_config_script.py tests/common/writers/test_auxiliary_tensor_writer.py tests/data/components/test_artifacts.py -q -p no:cacheprovider
```

- 概率：全部解析 arm 的精确归一化、有限预算质量守恒、跨深度累加、饱和 gate 的有限梯度。
- 数据：keyed catalog、checkpoint 身份、训练标定 split、因果 prefix state、trace schema、共享 writer 往返。
- 推断：有效唯一候选、预算耗尽、单 GPU 限制、平坦第一层分布下固定深度控制仍能完成候选。
- 装配：全部 9 arm 的 Hydra/T5 轻量实例化；脚本语法、引号、空值、非法参数、额外 override、两条队列的 54 条件覆盖。

此前加入既有 CGBS 模型与配置脚本回归的验收为 **159 passed**；与最终 93 项存在重叠，不应相加。此后修改集中在 MIR 搜索调度与 trace 校验，已由最终聚焦验收覆盖。

## 静态与规格验证

新增 Python 模块、测试及两处共享扩展通过 Ruff；OpenSpec strict 验证通过。两处共享扩展分别是可注入的辅助产物 validator 和 resolution/last-checkpoint artifact role，保留既有默认行为。

```powershell
openspec validate add-tiger-item-resolution --strict
git diff --check
```

## 交付与边界

- 54 次从头训练协议、两组 GPU 命令、单卡 inference/calibration、best/last 与 outcome diagnosis 命令见 [完整实验方案](experiment-plan.md)。
- 研究仓库已同步实施方案、机器状态、当前计划、执行看板、主方法设计更新与创新笔记；发布时校验原文件与目标内容 SHA256，保留既有实验结果。
- 不修改 TIGER baseline 主数据流；独立模块复用基础设施。
- 未启动真实训练、inference、calibration 或 diagnosis，没有新 W&B run；没有提交或推送 Git。
- CPU 检查不能证明真实 GPU 显存、DDP 吞吐、模型收敛或方法有效性，需由完整实验提供证据。
- COBRA/WIDE 属于明确记录差异的匹配适配控制，不代表官方复现；正式论文最近邻比较仍需履行对应复现与差异披露义务。
