## Why

在 `torchrun` 多进程下，每个 rank 都会实例化 data/model 并解析 `wandb://...` artifact 引用，当前实现会让每个 rank 都调用 W&B `artifact.download(...)`。这会产生重复下载日志、额外 API/网络开销，并可能在首次下载时并发写同一个 cache 目录。

## What Changes

- 在 W&B artifact 下载 helper 中加入分布式感知逻辑。
- 分布式已初始化时，仅 rank 0 调用 `artifact.download(...)`。
- 其他 rank 在 barrier 后从相同 download root 解析已下载文件。
- 非分布式运行保持现有行为。
- 逻辑集中在 `src/utils/wandb.py`，不改 launcher、writer、model 或 preprocessing 主链路。

## Capabilities

### New Capabilities

### Modified Capabilities
- `keyed-prediction-bundle-artifact`: W&B artifact 读取在共享 cache 的分布式运行中应只由 rank 0 执行下载。

## Impact

- Affected code:
  - `src/utils/wandb.py`
  - `tests/data/components/test_artifacts.py` or focused W&B helper tests
- No config migration is required.
- Single-node torchrun with shared local cache benefits directly.
- Cross-node runs still require a shared `wandb_cache_dir` or `GRID_WANDB_ARTIFACT_CACHE`.
