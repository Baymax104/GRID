# LETTER Toys 启动记录（2026-10-09）

## 当前阶段

2026-10-09 后续更新：三组 CF 训练与 CF 导出已完成并核验，固定命令和下游输入见 [CF 导出记录](letter-toys-cf-export-20261009.md)。下文保留初次启动快照。

在 node1 启动三个 seed 的正式 CF teacher 训练。Toys 尚无正式 CF、Tokenizer 或 SID 产物，因此当前执行依赖链的第一阶段；Tokenizer、CF 导出、SID 导出、推荐训练和 Testing 尚未执行。

## 运行与启动核验

核验时间：2026-10-09 21:19:56 UTC。以下为启动核验快照，后续进度以 W&B 和退出文件为准。

| Issue | Seed | 物理 GPU → 进程设备 | W&B run | 已记录步数 | train/loss |
| --- | --- | --- | --- | ---: | ---: |
| BMX-66 | 42 | GPU2 → CUDA0 | [wsflubzp](https://wandb.ai/baymaxam/GRID/runs/wsflubzp) | 549 | 1.223917 |
| BMX-67 | 200 | GPU4 → CUDA0 | [yix316dz](https://wandb.ai/baymaxam/GRID/runs/yix316dz) | 949 | 1.075395 |
| BMX-68 | 2026 | GPU5 → CUDA0 | [q7cblret](https://wandb.ai/baymaxam/GRID/runs/q7cblret) | 549 | 1.203160 |

三组状态均为 `running`，配置核验通过，未发现 Traceback、CUDA OOM 或退出文件。启动成功不代表 50k 步训练完成或最终实验有效。

正式协议：独立 LETTER 32 维 CF teacher，history=50，2 blocks，dropout=0.5，Adam lr=0.001、betas=[0.9, 0.98]，无 scheduler，单卡 batch=128、FP32；训练 50,000 步，每 1,000 步完整验证，按最大 `val/ndcg@10` 保存 best。使用 training split 更新梯度、evaluation split 选择 checkpoint，`ckpt_path=null`，`run_test_after_training=false`。

## 输入与代码来源

- 输入：`wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&alias=v7&file=merged_predictions_tensor.pt`。
- Artifact：`sem_embeds_inference-semantic-embedding:v7`；catalog 含 11,924 个商品。
- 运行前通过项目入口完成 Mutagen flush/status，三个 session 均为 Watching for changes，无 conflict。
- 三组运行的本地来源标记均为 verified；源码 SHA256 为 `087307c1d04142a606eabb8e8000c24407ce49c0580aef6f1648ce19b4079f6c`。
- 已核验运行时源码 archive 内每个文件的 size/SHA256，以及发布到 W&B 的源码 artifact 文件 digest 和来源元数据。
- Tokenizer 文件 SHA256 为 `aa8a4d0aaae9b086b53c3716dead27e20293d6249b28f4d64c94c8a18f59bf77`，包含 diversity sampling 性能修复。

## tmux 与记录

三个独立 tmux session 均有 `train` 和 `logs` 窗口：

- `letter_cf_bmx66_toys_s42_20261009-e94ie8w9`
- `letter_cf_bmx67_toys_s200_20261009-e94ie8w9`
- `letter_cf_bmx68_toys_s2026_20261009-e94ie8w9`

在 node1 查看 seed42：

```bash
tmux attach -t letter_cf_bmx66_toys_s42_20261009-e94ie8w9
```

node1 仓库根目录为 `/data3/weizhenyu/projects/GRID`。启动脚本、stdout/stderr 日志及退出码文件保存在 `logs/letter-toys-20261009-e94ie8w9/`，对应 `cf-bmx66`、`cf-bmx67`、`cf-bmx68`。训练结束后 wrapper 写入各自 `.exit` 文件。

完整启动命令与输出目录保存在 `training-plan.json`，启动核验结果保存在 `startup-verification.json`；这两份记录也已保存到本地同名目录。

后续阶段应使用正式 CF best checkpoint 导出的产物继续依赖链，并在推荐训练前核验完整 SID bundle。本次未建立后续阶段自动启动任务。
