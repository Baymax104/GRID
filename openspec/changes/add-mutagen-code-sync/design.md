## Context

本地 `E:\projects\GRID` 是代码的唯一可信源；远端 `node1:/data3/weizhenyu/projects/GRID` 的 Git 元数据陈旧且不得用于判定代码版本。远端同时保存约数十 GB 的数据集、预训练模型、虚拟环境、日志和实验产物，这些内容以远端为可信源，不能被同步工具扫描、回传或删除。现有成熟流程只需要把本地源代码、配置、启动脚本和依赖清单送到远端执行。

Mutagen 0.18.1 已安装在开发机，`node1` 已在用户 SSH config 中配置。正式同步会在用户 HOME 中创建 Mutagen daemon/agent 状态，但项目目录中的同步配置必须可审查、可重复。

## Goals / Non-Goals

**Goals:**

- 将本地受管代码以单向精确副本语义同步到远端对应路径。
- 用端点边界而非仅靠全仓库 ignore，确保远端重资产不可能被代码 session 删除。
- 将首次启动拆为“暂停创建”和“显式恢复”，允许在首次传输前检查 session。
- 提供稳定的 PowerShell 命令行入口和无需远端服务的轻量验证。

**Non-Goals:**

- 不同步或修复远端 `.git/`。
- 不在本地保存远端数据、模型、环境、日志或实验产物。
- 不自动安装/更新远端 `.venv`，也不自动运行训练、推理或 diagnosis。
- 不同步 `tests/`、`openspec/`、编辑器配置或本地临时目录。

## Decisions

### 使用四个窄范围 one-way-replica session

`src/`、`configs/` 和 `scripts/` 分别使用独立目录端点；根目录 session 只放行 `*.sh`、`pyproject.toml` 和 `uv.lock`。所有 session 都使用 `one-way-replica`，因此本地的修改、创建和删除是最终结果，远端对受管代码的漂移不会反向传播或阻塞同步。

不采用单个全仓库 replica，因为 ignore 配置错误可能触碰重资产；不采用 `one-way-safe`，因为远端代码明确不可信，远端漂移不应产生 conflict。

### 根目录采用 allowlist

根目录 session 先忽略所有根级条目，再取消忽略根目录 `*.sh`、`pyproject.toml` 和 `uv.lock`。根级目录因此不会被遍历，`.git/`、`data/`、`logs/`、`pretrained_models/`、`.venv/`、`wandb/`、`tmp/`、`tests/` 和 `openspec/` 均处于同步边界之外。

### 受管目录不保留远端运行时缓存

目录 session 不忽略 `__pycache__`、`*.pyc` 等运行时缓存。它们不属于远端重资产，且保留缓存会阻止 replica 删除本地已经移除、但远端仍含缓存的旧源码目录。远端执行产生的缓存允许被下一轮同步清理，以保证受管目录完整服从本地。

### 使用仓库级 mutagen.yml 和 PowerShell 管理入口

`mutagen.yml` 声明 session；`scripts/mutagen_sync.ps1` 固定从仓库根目录定位该文件，并映射 `start`、`status`、`flush`、`pause`、`resume`、`monitor`、`stop` 操作。所有项目命令使用 `--no-global-configuration`，避免用户全局 Mutagen 默认值改变安全契约。

`start` 始终使用 `--paused`，只创建 session 和部署所需 agent；`resume` 才允许开始传输。`monitor` 仅展示这四个具名 session。

### 不传播 Windows 权限

使用 Mutagen portable 权限默认行为，不尝试把 Windows ACL 或执行位复制到 Linux。GRID 启动脚本按既有约定通过 `bash <script>.sh` 执行，新文件不依赖 POSIX executable bit。

### 在 AGENTS.md 固化操作门禁

`AGENTS.md` 内联可信源、受管范围和完成标准，并指向 `mutagen.yml` 与管理脚本获取可执行细节。这样后续 agent 会在远端实验前主动 `flush` 并检查 Watching/conflict 状态，同时避免复制一份容易与配置漂移的完整参数表。

## Risks / Trade-offs

- [根目录 allowlist 写错可能扩大范围] → 用静态测试精确断言 allowlist，并在首次启动时保持 session 暂停。
- [one-way-replica 会删除远端受管目录中的额外代码] → 这是本地唯一可信源契约的预期行为；重资产不放在这些端点中。
- [同步会清理受管目录内的 Python 缓存] → 缓存不作为运行资产保存；长任务依赖已加载代码，不以缓存文件持久存在作为契约。
- [远端运行期间同步 Python 文件可能形成短暂版本混合] → 在启动长任务前执行 `flush`；运行中的任务不以热更新作为一致性保证。
- [Mutagen project 功能在 0.18.1 中标记为 Experimental] → 配置保持简单，并通过封装脚本集中命令；必要时可等价迁移为四条 `mutagen sync create` 命令。
- [首次 start 会写入用户 HOME 中的 daemon/agent 状态] → 这是 Mutagen 正常工作所需，项目重资产目录仍不受影响。

## Migration Plan

1. 提交并验证 `mutagen.yml`、管理脚本和静态测试。
2. 执行 `scripts/mutagen_sync.ps1 start`，以暂停状态创建四个 session。
3. 执行 `status` 检查端点、模式和忽略规则。
4. 执行 `resume` 完成首次代码同步，再用 `flush` 作为运行实验前的同步门禁。
5. 如需回滚，执行 `stop` 终止项目 session；远端重资产和 Git 元数据不受影响。

## Open Questions

无。
