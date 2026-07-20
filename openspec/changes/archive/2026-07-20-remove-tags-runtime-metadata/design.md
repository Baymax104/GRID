## Context

`tags` 在当前仓库中的作用完全属于运行元信息层：如果配置里没有 `tags`，`extras(cfg)` 会调用 `enforce_tags()` 进行交互式补全，并把结果写入 Hydra 输出目录的 `tags.log`；另外 `log_hyperparameters()` 会将 `tags` 作为 hparams 一部分发送给 logger。它不参与 train / inference 分流，不影响 datamodule、model、trainer 或 callback 的业务行为。

本次目标是把这套没有实际消费场景的元信息接口彻底移除，而不是仅仅关闭“强制标签”行为。

## Goals / Non-Goals

**Goals:**
- 删除 experiment 顶层 `tags` 字段。
- 删除 `extras.enforce_tags` 及其运行前补全逻辑。
- 删除 `tags.log` 产物。
- 停止在 logger hparams 中发送 `tags`。

**Non-Goals:**
- 不改变 `task_name`、`print_config`、`config_tree.log` 等其他运行元信息机制。
- 不改变 W&B logger 本身的启用方式。
- 不重构 Hydra 输出目录结构。

## Decisions

### 1. 完整移除 `tags` 体系，而不是仅关闭强制检查
- 决策：删除 experiment 顶层 `tags` 与 `extras.enforce_tags`，同时移除相关代码逻辑与日志输出。
- 原因：如果标签本身没有消费场景，仅关闭交互检查仍会保留一套无用配置接口。

### 2. 删除 `tags.log`
- 决策：运行输出目录不再生成 `tags.log`。
- 原因：该文件只承载待删除的标签元信息，保留没有意义。

### 3. 停止向 logger hyperparameters 发送 `tags`
- 决策：`log_hyperparameters()` 不再包含 `hparams["tags"]`。
- 原因：既然配置层已删除 `tags`，logger 也不应再保留其历史接口。

### 4. 修正规格中对 `tags.log` 的保留描述
- 决策：更新 `add-cli-dry-run` 的相关 spec/proposal/design，使 dry run 保留的元信息文件只包含仍真实存在的内容。
- 原因：当前规格仍声明 dry run 可以保留 `tags.log`，这会与本次实现目标冲突。

## Risks / Trade-offs

- [历史命令或私人配置仍传入 `tags=` 覆盖] → 会失去兼容性，但这是用户接受的 breaking change。
- [历史文档仍提到 `tags.log`] → 需要同步清理，否则会制造错误预期。
- [W&B 中失去一项 hparam 元信息] → 这是有意删减，不影响监控主路径。

## Migration Plan

1. 删除配置层的 `tags` 与 `enforce_tags`。
2. 删除运行前标签补全逻辑、`tags.log` 写入和 logger hparams 中的 `tags`。
3. 更新实验配置、AGENTS.md 和相关 OpenSpec 文档。
4. 做最小静态检查与全文搜索，确认仓库不再依赖 `tags` 体系。

## Open Questions

- 当前无阻塞性开放问题。
