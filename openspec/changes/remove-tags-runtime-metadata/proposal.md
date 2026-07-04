## Why

当前仓库中的 `tags` 配置与 `extras.enforce_tags` 逻辑只用于补充运行元信息：强制用户提供标签、写出 `tags.log`，并将 `tags` 附加到 logger 的 hyperparameters。对于主要依赖 W&B 进行监控、且没有消费标签场景的使用方式，这套机制增加了配置长度、交互行为和额外日志文件，但不提供实际价值。

现在需要移除 `tags` 体系及相关运行时逻辑，让 experiment 配置和运行前置流程更简洁。

## What Changes

- 删除所有 official experiments 顶层 `tags` 配置。
- 删除 `extras.enforce_tags` 配置项及其运行前校验/交互逻辑。
- 删除 `tags.log` 写入逻辑。
- 停止向 logger hyperparameters 发送 `tags` 字段。
- 同步更新受影响的文档与 OpenSpec 说明。

## Capabilities

### Modified Capabilities
- `cli-dry-run`: dry run 保留的运行元信息输出将不再包含 `tags.log`。

## Impact

- 受影响代码：`src/utils/utils.py`、`src/utils/rich_utils.py`、`src/utils/logging_utils.py`、`src/utils/__init__.py`
- 受影响配置：`configs/extras/default.yaml`、所有 `configs/experiment/*.yaml`
- 受影响文档：`AGENTS.md` 与所有提及 `tags` / `tags.log` / `enforce_tags` 的说明
- **BREAKING**：官方运行配置不再支持顶层 `tags`，也不再支持 `extras.enforce_tags`
