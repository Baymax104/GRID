## Why

BMX-69 的配置和产物可核验，但缺少运行时源码快照，无法完整还原未提交修改。正式运行需要自动留存实际源码，避免只依赖 commit 或远端不可信 Git 状态。

## What Changes

- 统一入口在正式运行开始前生成源码归档、SHA256 manifest 和环境信息。
- 启用 W&B 时上传真实文件为 code Artifact，将指纹写入 resolved config。
- 本地同步入口生成可信 Git 来源记录；运行端仅在源码指纹匹配时引用其 commit/dirty 信息。
- dry-run 不生成或上传快照；旧 run 不事后伪装补录。

## Capabilities

### New Capabilities

- `run-source-snapshot`: 运行源码、环境及可信本地来源的留档契约。

### Modified Capabilities

无。

## Impact

涉及统一 launcher、utils 快照工具、共享 writer、全局 component config 与 Mutagen 本地入口；不增加依赖，不修改实验算法、数据或预算。
