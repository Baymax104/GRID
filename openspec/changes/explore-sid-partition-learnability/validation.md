# 验证记录

日期：2026-09-18。完成首阶段实现，未运行真实数据、模型训练或GPU推理。

## 已通过

- 42项测试：合成正向/无信号控制、严格重放、未见上下文回退、fit/eval分离、用户前缀去重、输入预算、目录身份、三/四列SID、Lightning batch传输、Hydra compose及输出目录解析、dry-run writer/logger禁用、Bash语法/quoting/空值/非法参数/override顺序，以及共享writer和metric callback回归。
- `openspec validate explore-sid-partition-learnability --strict`。
- 改动Python文件Ruff检查。
- research-state.yaml解析、协议文件链接、未启动实验状态检查。
- `git diff --check -- src/utils/launcher.py`。

测试命令（从GRID根目录，basetemp应使用新的任务临时目录）：

```powershell
uv run pytest tests/quantization/test_sid_partition_probe.py tests/test_sid_partition_probe_config_script.py tests/utils/test_metric_callback_attachment.py tests/common/writers/test_structured_analysis_writer.py --basetemp=tmp/sid-probe-tests-03 -p no:cacheprovider -q -rs
```

本机实际添加`--cache-dir E:\projects\GRID\tmp\uv-cache`以避开外部uv缓存权限；Windows沙箱内pytest临时目录及Git Bash不可用，经过自动审批后在沙箱外完成同一合成测试，最终42 passed、0 skipped。未放宽数据测试边界。4项第三方DeprecationWarning不影响结果。

## 未验证范围

- 真实training记录是否前缀兼容、支持组是否足够、真实统计信号与耗时。
- W&B真实下载、运行及发布；仅复用已存在的loader/writer/lineage并测试本地输入和共享组件。
- 远端代码同步；本轮未flush，不报告远端已就绪。
- 后续量化映射干预与推荐模型有效性。E2/E3只设计，尚未实现。

所有真实运行仍由用户手动启动。执行协议见research/docs/2026-09-18-sid-partition-exploration-protocol.md。
