# 验证记录

日期：2026-09-18。仅合成输入与配置/脚本验证；未运行真实 E2 准备、训练、预测或审计。

## 聚焦验证

最终命令从 GRID 根目录执行：

```powershell
uv --cache-dir E:/projects/GRID/tmp/uv-cache run --no-sync pytest tests/quantization/test_sid_partition_intervention.py tests/data/components/test_sid_partition_intervention_inputs.py tests/test_sid_partition_intervention_config.py tests/recommendation/test_a_score_search_audit.py tests/common/writers/test_structured_analysis_writer.py tests/quantization/test_sid_partition_probe.py tests/test_sid_partition_probe_config_script.py -p no:cacheprovider --basetemp=tmp/e2-tests-04 -q -rs
```

结果：**102 passed，0 skipped，4个第三方deprecation warnings，9.96秒**。沙箱内 Git Bash 不可用的15项跳过，已在获准环境中完整补跑；没有把跳过当通过。早期测试同名冲突通过重命名新增测试解决；负对照fixture修正为原组静态伙伴后检验到预期拒绝，不修改真实判定阈值。

覆盖：

- 稀疏计数与 E1 稠密公式一致，包括未见context回退。
- 合成行为正对照输出三臂；反转internal check信号后停止，但候选选择与映射指纹不变。
- 完整tuple、四层全部前缀占用、同改动item集合、不同首组对照；固定候选和几何门槛。
- 支持不足、dry-run、协议偏离均不导出可训练bundle；非法配置拒绝。
- E1未通过或记录指纹不符拒绝；训练映射、内容或训练文件指纹不同拒绝；续训拒绝。
- shared writer keyed bundle roundtrip、重复文件名/非法路径/行数/键类型失败原子性，以及原JSON/CSV发布回归。
- 四个 experiment 的 Hydra compose、随机mask_ce、20k/500、NDCG checkpoint、禁自动testing、guard装配与dry-run禁写。
- Bash语法、两种参数形式、空notes、特殊字符quoting、非法参数、显式dry-run及额外override最后优先。
- 新mask_ce目录审计加载完整匹配checkpoint，验证归一化精确概率/beam证据；原A入口与validator仍拒绝mask_ce默认输入。

Ruff 对本轮源码及测试检查通过。OpenSpec strict 校验通过。研究状态 YAML 和运行文档链接另外做轻量检查。

## 运行与同步边界

已通过规定入口执行 Mutagen flush；其后四个session均 Watching for changes、无conflict，端点为本地GRID至node1受管代码目录。没有修改远端Git、数据、模型、依赖环境或实验产物，也没有新W&B run。同步快照在本地 tmp/e2-mutagen-status.txt。

实际数据匹配数量、CPU准备耗时、internal check转移、推荐增益均未知。下一步由用户手动运行一次CPU准备，按输出门槛决定是否启动三臂；完整协议及命令见 [研究文档](../../../../research/docs/2026-09-18-sid-partition-e2-protocol.md)。
