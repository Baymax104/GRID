# CoPMRec v3.1 单卡推理

最新状态：用户已完成推理 `zznw291t`，已独立核验完整 Testing。高内容遗漏114→81，最终净新增命中11，收益区间跨零；首层恢复保留，整体暂不晋级。见 [结果报告](copmrec-v3-1-testing-zznw291t-result.md)。下文为本次已完成推理的配置及命令留档，不表示建议再次运行。

日期：2026-10-04。用户授权实施此前提出的 root 包络修正并准备推理命令。复用 v3 训练 run `rbha00vx` 的 validation-selected best43500；解码版本为 v3.1，checkpoint 训练版本仍为 v3，无重新训练。

## 改动与固定条件

首层仍按 v3 Max 混合概率选择相同 beam20。设原 Max/Mass 根层混合 log 概率为 rX/rM，root 路径先验为 `rE = maximum(rX,rM) - logsumexp(maximum(rX,rM))`。第二个 SID 决策对该 root 所有合法孩子一次性增加 `rE-rX`；后续局部 Mass 概率不变，终排继续使用同一内容分数，保留全部 cold 候选。

同一 SID/embedding、seed42、beam20、Top10、predict batch32、full testing。physical GPU2 映射到 logical GPU0；单进程。新 trace schema `liger_paths_v2_root_envelope` 分开保存原局部条件概率、root 先验/补正、搜索增量与累计分数。

固定 checkpoint：`copmrec_beauty_v3_train-checkpoint:v0`，producer `rbha00vx`，Artifact digest `48204b0d67bf2ba6c244d6f1b09d583a`，文件 MD5 base64 `hO6s8wKBi1kK3MD2hmR6OQ==`，学习 alpha `0.668786346912384`。选点及输入来源已在 [v3审计](copmrec-v3-rbha00vx-testing.md) 和 [v3结果](copmrec-v3-testing-zy8946l2-result.md) 核验。

## 单卡命令

在 node1 执行：

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

COPMREC_V3_BEST_CKPT='wandb://baymaxam/GRID/rbha00vx?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=043500.ckpt'

CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash copmrec_v3_1_inference.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint "$COPMREC_V3_BEST_CKPT" \
  --devices '[0]' \
  --seed 42 \
  --master-port 29549 \
  --group copmrec_v3_1_rbha00vx_testing \
  --notes 'CoPMRec v3.1 testing：复用rbha00vx best43500；首层Max原选择，第二层一次root Max/Mass包络路径补正，后续Mass；单beam20加全部cold、content终排；与zy8946l2比较恢复及损失；Testing已用于开发分析'
```

配置默认开启 candidate/path trace，使用专用 writer validator，无需附加 callbacks override。正式输出由共享 writer 保存并按现有规则发布，实际运行字节通过默认 source snapshot 留档。`--dry-run` 仅适合显式 smoke，不加到正式命令。

## 验证与结果门禁

比较既有 v3 推理 `zy8946l2`：第一层 frontier 逐项相同，dense 分数/排名与输入身份一致；检查 114 个遗漏目标的恢复、103 个原增量命中的损失、原首层恢复案例的保留率及全体新增/丢失命中和共同命中位置变化，再报告 Recall/NDCG 与配对区间。

新增计划为一次用户手动推理、零训练。正向净改善支持此 decoder 的开发方向；负向净损失或失败移层停止当前补正形式，保留首层 Max；不确定结果不自动追加扫描或重置旧预算。Testing 已用于开发分析，此次不构成独立确认。效果仍待完整推理。

实施与验证证据见 [verification.json](evidence/copmrec-v3-1-implementation-20261004/verification.json)。本轮没有启动完整训练或推理，未写入 W&B/Linear。
