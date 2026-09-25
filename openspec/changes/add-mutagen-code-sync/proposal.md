## Why

GRID 当前依赖人工复制本地代码到 `node1`，缺少可重复、可审计且不会触碰远端重资产的命令行同步契约。需要用 Mutagen 固化“本地 Git 工作区是代码唯一可信源、远端是数据和运行环境唯一可信源”的既有工作流。

## What Changes

- 新增仓库级 Mutagen project 配置，以多个窄范围 session 将本地代码单向复制到 `node1:/data3/weizhenyu/projects/GRID`。
- 对 `src/`、`configs/`、`scripts/`、根目录 `*.sh`、`pyproject.toml` 和 `uv.lock` 使用本地优先的精确副本语义。
- 将远端 `.git/`、`data/`、`pretrained_models/`、虚拟环境、日志、W&B 文件和其他实验重资产置于同步边界之外。
- 提供 PowerShell 命令行入口，用于启动、查看、强制刷新、暂停、恢复和终止项目同步；启动流程默认先创建暂停的 session，并要求显式恢复后才传输文件。
- 受管目录采用完整 replica 语义，包括清理远端遗留的源码缓存和仅远端存在的旧代码目录。
- 将可信源、同步边界、首次启动门禁和实验前完成条件写入项目 `AGENTS.md`，约束后续 agent 操作。
- 增加配置与命令行入口的轻量测试，不依赖真实 SSH、Mutagen daemon 或远端文件系统。

## Capabilities

### New Capabilities

- `mutagen-code-sync`: 定义 GRID 本地代码到远端运行目录的可信源、同步范围、安全门禁和命令行操作契约。

### Modified Capabilities

无。

## Impact

- 新增仓库级 Mutagen 配置、PowerShell 管理脚本及其测试。
- 开发机需要已安装 Mutagen 0.18.x，并在 SSH config 中配置 `node1`。
- 不新增 Python 运行时依赖，不修改训练、推理或 diagnosis 入口。
- 首次实际启动会在本地和远端用户 HOME 中创建 Mutagen daemon/agent 状态，但不会把远端重资产纳入同步。
