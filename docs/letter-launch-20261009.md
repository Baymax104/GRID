# LETTER 修复后推荐训练启动记录

日期：2026-10-09（Asia/Shanghai）。用户明确授权在 node1 启动训练，启动 Beauty/Sports × seed42/200/2026 六组推荐训练。前缀约束修复见 [修复记录](letter-prefix-fix-20261009.md)。

## 训练协议

从随机初始化重新训练，`ckpt_path=null`，不恢复已删除的旧 run。沿用原六组已核验 SID、双卡 DDP、每卡 batch128/global256、FP32、history20、temperature1、beam20/top10、50k optimizer steps、每500步完整 evaluation、val/ndcg@10 最大选 best；`run_test_after_training=false`。

物理 GPU0/1、2/3、4/5 分别映射到各进程组内 CUDA0/1，同一 GPU pair 承载一个 Beauty 和一个 Sports 实验。GPU2/3 启动前有其他任务共享，未停止或修改其他任务。新端口为 29760–29765。

所有训练通过根目录 `letter_train.sh` → `uv run torchrun --nproc_per_node=2 -m src.main`，`UV_NO_SYNC=1`，没有修改虚拟环境。

## 启动核验快照

| Issue | Dataset / seed | 物理 GPU | 新 W&B run | 已记录 global_step | train/loss |
| --- | --- | --- | --- | ---: | ---: |
| BMX-60 | Beauty / 42 | 0,1 | [diiyhjux](https://wandb.ai/baymaxam/GRID/runs/diiyhjux) | 299 | 7.109196 |
| BMX-61 | Beauty / 200 | 2,3 | [qzcesmm6](https://wandb.ai/baymaxam/GRID/runs/qzcesmm6) | 299 | 6.825863 |
| BMX-62 | Beauty / 2026 | 4,5 | [yfdhrh8g](https://wandb.ai/baymaxam/GRID/runs/yfdhrh8g) | 349 | 6.631340 |
| BMX-63 | Sports / 42 | 0,1 | [0gokpcne](https://wandb.ai/baymaxam/GRID/runs/0gokpcne) | 299 | 7.157301 |
| BMX-64 | Sports / 200 | 2,3 | [wdmgy3lu](https://wandb.ai/baymaxam/GRID/runs/wdmgy3lu) | 299 | 7.177821 |
| BMX-65 | Sports / 2026 | 4,5 | [ktf4xvx1](https://wandb.ai/baymaxam/GRID/runs/ktf4xvx1) | 299 | 7.453630 |

六组 W&B state=running，DDP 两个 rank 已注册、loss 有限、未发现 traceback/OOM 或退出文件。resolved config 中 seed、SID、训练预算、batch、beam 与不自动 Testing 均核验通过；消费的 SID Artifact 分别为 Beauty v0/v1/v2、Sports v0/v1/v2。

Mutagen flush 后三个 session 均 Watching for changes，无 conflict。六组运行端源码 SHA256 均为 `5f7b6a1590d12f3dce29072976d202e694533d1e16c105413d9d78e7306e4075`，origin=verified。逐文件核验 source archive 的大小/SHA256，确认 backbone SHA256 为修复版本 `ca54930ef8739baf9b770910dfb8b5c86d32676b3cae62f66cffb05f42daabd4`。每个 `grid-source-<run_id>:v0` Artifact 均 COMMITTED，三个发布文件的 manifest digest 与运行端文件一致。

这是启动快照，不代表50k完成、全量验证性能或最终基线有效性。

## tmux 与证据

在 node1 执行：

```bash
tmux attach -t letter_bmx60_beauty_s42_20261009-prefixfix-aad918p3
```

其他 session 使用相同后缀 `20261009-prefixfix-aad918p3`，前缀分别为 `letter_bmx61_beauty_s200`、`letter_bmx62_beauty_s2026`、`letter_bmx63_sports_s42`、`letter_bmx64_sports_s200`、`letter_bmx65_sports_s2026`。每个 session 包含 train 与 logs 窗口，Ctrl-b d 脱离后继续训练。

node1 记录目录：`/data3/weizhenyu/projects/GRID/logs/letter-formal-20261009-prefixfix-aad918p3/`。每组 `train-bmxNN.sh` 保存完整命令，`.log` 保存日志，结束后写入 `.exit`。`training-plan.json` 保存新旧 run 对应关系、SID URI、GPU/端口和目录；`training-startup-verification.json` 保存逐组核验与 UTC 时间戳。两个 JSON 已复制到本地同名 logs 目录。
