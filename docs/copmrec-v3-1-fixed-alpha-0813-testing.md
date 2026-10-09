# v3.1 固定推理 alpha=0.813：单卡对照

最新状态：用户已完成 `3lb22b0r`，完整Testing独立核验通过。高内容遗漏81→52，超dense增量81→52，新增/损失各40、Recall不变，NDCG10+0.0775%但区间跨零；默认保留学习alpha，固定设置不晋级。见 [结果报告](copmrec-v3-1-alpha0813-testing-3lb22b0r-result.md)。下方命令为已完成对照留档，不建议重复运行。

日期：2026-10-04。用户明确要求在保留的 v3.1 上固定 alpha=0.813 看效果。仅增加推理覆盖，复用 rbha00vx 的 validation-selected best43500，不训练、不修改 checkpoint 学习参数。本次单独授权一次用户手动完整推理，不扫描其他权重，不实施此前候选完成上界方案。

## 对照问题与有效性

当前 v3.1 的 zznw291t 使用 checkpoint 学习 alpha=0.668786346912384。固定0.813的内容权重可在同一模型下检验：较强内容引导是否减少强内容目标遗漏，并取得最终净收益。选择0.813来自用户指定近似 v0 权重；它不等于重新训练出的 v0，也不是最优值结论。已有同 checkpoint 内容引导对照支持剪枝前使用内容，尚未证明更高权重必然更好。

首层仍用 Max，后续仍用 Mass，root Max/Mass 包络仍在第二个SID决策补正一次，所有混合均使用同一个实际alpha=0.813。alpha变化允许首层root集合改变，不要求与zznw291t frontier相同；其余内容/生成参数、dense logits及排名、输入、用户、目录、cold并集、seed42、FP32、beam20、content Top10终排必须保持。

配置使用已有 `model.root.inference_mixture_alpha` 字段，null仍读取checkpoint；有限[0,1]覆盖只由v3.1推理类接受，旧v3训练/推理契约不改。state_dict无新增参数，训练接口继续拒绝。trace同时记录实际 `content_mixture_alpha=0.813`、`mixture_alpha_source=fixed_inference`、`checkpoint_mixture_alpha≈0.668786`，保持原root-envelope schema/writer。

## 单卡完整推理命令

在 node1 仓库根目录 `/data3/weizhenyu/projects/GRID` 执行。物理GPU2映射logical GPU0，单进程，每卡predict batch32，full Testing。根脚本使用统一 `uv run -m src.main`，默认 source snapshot 归档实际运行字节。

```bash
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
  --group copmrec_v3_1_alpha0813_rbha00vx_testing \
  --notes 'CoPMRec v3.1 fixed-alpha对照：复用rbha00vx best43500，推理alpha精确固定0.813；首层Max、第二层一次root包络补正、后续Mass；beam20加全部cold、content终排；对照zznw291t学习alpha0.668786，检查最终净收益与遗漏/增量损失；Testing已用于开发分析' \
  model.root.inference_mixture_alpha=0.813
```

## 结果判断

实施检查已通过：63项聚焦测试及ruff，覆盖固定概率枚举、四层HF transition score、checkpoint与dense分数不变、trace/标签隔离、Hydra与脚本透传及旧v3回归；OpenSpec严格验证通过。Mutagen flush成功，三个会话均Watching for changes且无冲突。node1核验7个运行文件SHA256、实际checkpoint摘要及164个state tensor严格加载；命令mock与Hydra compose、shell语法均通过。远端核验无模型前向或decoder search，不构成效果实验。

先核对实际checkpoint、源码、输入、全体用户和trace实际权重；独立复算指标，配对比较zznw291t。主要看NDCG10，同时看Recall10、原81个高内容遗漏、原81个超dense增量命中的保留及全体新增/损失、原首层恢复47候选/29命中的保留；首层集合变化本身不是有效性失败。

正向净改善支持该固定权重解码设置，但不能由此证明content终排是唯一原因或0.813最优；负向净损失回退学习值；局部恢复而损失抵消、或区间跨零，记录不确定，不自动扫描或扩训练预算。原v3.1学习权重仍保留为对照基座，Testing开发边界保持。

实施验证和同步状态见 [verification.json](evidence/copmrec-v3-1-alpha0813-20261004/verification.json)。本轮完整训练/推理启动0、W&B/Linear写入0，完整效果待用户运行。
