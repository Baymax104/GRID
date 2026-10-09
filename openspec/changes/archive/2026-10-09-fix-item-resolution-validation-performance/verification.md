# Validation 性能修复验收

日期：2026-09-14。用户已告知停止原两条运行并删除对应 W&B run；本轮未访问、恢复或删除远端运行。

## 实现

- `eval_step` 与普通 `generate` 使用 `collect_trace=False`；`predict_step` 根据正式 trace 开关选择路径。解析 arm 轻量路径不分配事件 trace、不计算概率上界/证书、不为证据计时执行 CUDA synchronize。
- 完整 trace 通过静态 item 祖先索引批量计算上界。静态 count/depth 用 CPU 元数据，gate 按展开批次取回，消除最终证书的逐节点 GPU 成员查询。
- 祖先索引为 non-persistent buffer，不改变 checkpoint 字段及模型 contract。
- 训练目标、搜索调度、Q=64/S=4096、验证频率500、验证集及原 54 条件矩阵不变。

## 自动检查

```powershell
uv --cache-dir tmp/uv-cache run --no-sync pytest tests/recommendation/test_tiger_item_resolution.py tests/test_tiger_item_resolution_config_script.py tests/common/writers/test_auxiliary_tensor_writer.py tests/data/components/test_artifacts.py -q -x -p no:cacheprovider
```

**105 passed**，4 条依赖弃用提示。覆盖全部 9 arm 轻量/完整输出一致、eval/predict 接线、混合层/空前沿证书参考、宽目录查询次数、checkpoint 兼容，以及既有概率、梯度、schema、writer、配置和脚本检查。

Ruff、OpenSpec strict 与 `git diff --check` 通过。没有启动完整训练、inference 或 diagnosis。

## 目录规模 CPU 对比

命令：`uv --cache-dir tmp/uv-cache run --no-sync python -m tmp.benchmark_mir_performance_fix`。

使用既有 Beauty raw SID 拓扑（12101 items）、合成内容、未训练小 T5、CPU 单线程、batch16；原实现从修复前文件备份加载。三次交替测量，不含模型初始化或训练。

| 搜索路径 | 中位秒/批 | members 查询次数 |
|---|---:|---:|
| 修复前完整搜索 | 4.6956 | 32748 |
| 修复后完整 trace | 0.3051 | 8 |
| 修复后 validation | 0.2938 | 8 |

完整 trace 约 15.39 倍，validation 约 15.98 倍；三次都严格比较推荐 SID/分数，完整 trace 的所有非计时张量均与原实现相同。原始记录：`tmp/mir_validation_performance_fix_2026-09-14.json`。

这是 CPU 搜索性能数据，不能作为真实 GPU 或整轮 validation 加速比。真实四层 T5、DDP、数据读入及服务器 GPU 吞吐尚未测量。

## 重启

将本次代码同步到服务器后，从 GRID 根目录运行原有两条队列命令。原实验已停止且 run 已删除，本轮从头开始，不使用旧 run URI 恢复。所有 checkpoint 推理仍为单 GPU。

```bash
# 终端一：GPU 0、1
bash ./tiger_item_resolution_suite.sh \
  --queue 1 \
  --beauty-data-dir data/beauty \
  --sports-data-dir data/sports \
  --notes "MIR v1 full method comparison; validation performance fix; queue 1"
```

```bash
# 终端二：GPU 2、3
bash ./tiger_item_resolution_suite.sh \
  --queue 2 \
  --beauty-data-dir data/beauty \
  --sports-data-dir data/sports \
  --notes "MIR v1 full method comparison; validation performance fix; queue 2"
```

第一轮 validation 完成后，应能继续超过 step500 并产生 `val/ndcg@10`；实际耗时由新 run 记录确认。完整队列仍为每组27次训练。研究方案选择门槛与 `mir-v1` 保持一致。
