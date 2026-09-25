# 冻结checkpoint推理验证

## 范围与来源

只运行first_only和deep_residual两个新条件，使用训练时选择的最佳checkpoint。模型模式、码本来源与训练契约保持一致；输出逐用户预测和prefix trace。

A复用W&B run `cgbs-eval-5g3wpbg7-1789305926`：已通过只读API确认finished、evaluation、beam10，使用A checkpoint digest `aaba5b47dc9a5e94c0d23b75cf697238`，输出recommendation digest `b8dfc94e8d79a939ba204f25986fa311`及trace digest `095868b7b154b2b7a8ba58aa8a95d346`。新结果完成后还必须按用户key、完整标签和目录身份对齐，不能只比较聚合指标。

first_only：run46kuqs6m，19500步，checkpoint digest `4f2fe5145446eee3bbdeb6089b00f043`。
deep_residual：runnq8he993，19000步，checkpoint digest `a5fad97f3a5ae19ce8adb9d49de15457`。

## 手动启动

在node1仓库根目录`/data3/weizhenyu/projects/GRID`的两个终端分别执行。每个任务只需一张卡；示例选用本轮实际训练卡组中的4和6。若卡占用，可修改CUDA_VISIBLE_DEVICES为其他空闲卡，devices仍为[0]。

```bash
CUDA_VISIBLE_DEVICES=4 NPROC_PER_NODE=1 bash ./tiger_catalog_grounded_inference.sh \
  --data-dir data/beauty --dataset beauty --data-split evaluation \
  --semantic-id-path 'wandb://baymaxam/GRID/4vyi4o6w?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint-path 'wandb://baymaxam/GRID/46kuqs6m?role=checkpoint&file=checkpoint_epoch=000_step=019500.ckpt' \
  --arm token_content_init --group rkmeans --devices '[0]' \
  --beam-width 10 --seed 42 --master-port 29760 \
  --notes 'Frozen first-only checkpoint; paired evaluation against A; no training' \
  model=tiger_content_initialization_inference \
  task_name=tiger_content_initialization_first_only_beauty_evaluation_beam10 \
  +initialization_protocol=matched-code-geometry-v1 \
  +initialization_condition=first_only \
  +initialization_reference_run=5g3wpbg7
```

```bash
CUDA_VISIBLE_DEVICES=6 NPROC_PER_NODE=1 bash ./tiger_catalog_grounded_inference.sh \
  --data-dir data/beauty --dataset beauty --data-split evaluation \
  --semantic-id-path 'wandb://baymaxam/GRID/4vyi4o6w?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint-path 'wandb://baymaxam/GRID/nq8he993?role=checkpoint&file=checkpoint_epoch=000_step=019000.ckpt' \
  --arm token_content_init --group rkmeans --devices '[0]' \
  --beam-width 10 --seed 42 --master-port 29761 \
  --notes 'Frozen matched-residual checkpoint; paired evaluation against A; no training' \
  model=tiger_residual_initialization_inference \
  task_name=tiger_content_initialization_deep_residual_beauty_evaluation_beam10 \
  +initialization_protocol=matched-code-geometry-v1 \
  +initialization_condition=deep_residual \
  +initialization_reference_run=5g3wpbg7
```

两任务使用不同通信端口和task_name。单卡推理不改变训练batch或模型，避免为这一步额外引入DDP输出合并差异。残差配置启动时仍需读取同一量化器以验证初始化契约，随后严格加载已训练推荐模型权重；该过程不会训练量化器或推荐模型。

## 预先固定的分析

首先核对预测/trace完整性、checkpoint和SID/content lineage，再对齐用户key与标签。主指标NDCG@10，辅助Hit/Recall@10、前缀存活、命中增失和排名迁移；配对prefix-cluster bootstrap估计A相对两个对照的差异区间，不把相邻训练点当作独立样本。

这里evaluation已经用于训练checkpoint选择，因此属于选定模型在相同评估集上的配对检查，不能宣称独立测试集泛化或跨seed显著性。用户层配对分析也不能替代未来独立seed复核。不自动追加训练、推理网格或新模块。

## 交付验证

配置/脚本与初始化单测143 passed，另有2项固定checkpoint URI quoting测试通过。覆盖训练→推理契约一致、state_dict严格加载、用户key、evaluation输入、prefix trace以及metrics=null。未启动真实Trainer推理；现有脚本未改动，继续沿用其语法/参数回归验证。

OpenSpec严格验证通过。Mutagen flush成功，随后四个session均Watching for changes且无冲突；两份新增推理配置已同步到node1。未启动实验、未创建Git提交。
