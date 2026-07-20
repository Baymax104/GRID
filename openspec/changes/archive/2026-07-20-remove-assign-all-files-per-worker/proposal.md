## Why

当前数据加载配置仍保留 `assign_all_files_per_worker` 这一训练期特殊策略，但基于仓库现有 Amazon beauty / toys / sports 数据分片情况，该策略并非主线实验的必要条件。它引入了额外的配置理解成本和 datamodule / dataset 分支逻辑，使新增实验时更难判断哪些参数是真正需要关心的。

现在需要明确收缩官方能力：删除 `assign_all_files_per_worker` 配置项及相关运行逻辑，让文件分配语义统一回到默认的 worker 间分摊处理，降低实验配置复杂度。

## What Changes

- **BREAKING** 删除 dataloader 配置中的 `assign_all_files_per_worker` 语义，不再允许通过官方 experiment 开启“每个 worker 读取全部文件”策略。
- **BREAKING** 删除 datamodule / dataset / 文件分配工具中围绕 `assign_all_files_per_worker` 的分支与约束逻辑。
- 更新量化训练 experiment 配置，移除当前残留的 `assign_all_files_per_worker: true`。
- 清理相关注释和文档，使官方数据加载路径只表达按 worker 正常分摊文件的语义。

## Capabilities

### New Capabilities
- `worker-file-assignment-simplification`: 官方数据加载配置不再暴露“全部 worker 共享全部文件”策略，统一采用标准文件分摊语义。

### Modified Capabilities

## Impact

- 受影响代码：`src/data/loading/datamodules/`、`src/data/loading/components/dataloading.py`、`src/data/loading/components/interfaces.py`、`src/data/loading/utils.py`
- 受影响配置：当前声明 `assign_all_files_per_worker` 的量化训练 experiment
- 不引入新依赖，但属于配置能力收缩与数据加载逻辑简化
