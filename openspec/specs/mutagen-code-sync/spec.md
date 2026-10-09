# mutagen-code-sync Specification

## Purpose
定义本地代码向 node1 单向精确同步的白名单、三个窄范围端点和生命周期入口，保护远端重资产并阻止反向传播，规定首次暂停启动及 flush 完成门禁，使同步结果可验证。
## Requirements
### Requirement: 本地代码是唯一可信源
系统 SHALL 将本地受管代码作为同步结果的唯一可信源，并以单向精确副本语义传播本地的创建、修改和删除。

#### Scenario: 本地代码发生变化
- **WHEN** 用户修改、创建或删除本地受管代码并触发同步
- **THEN** 远端对应受管路径 SHALL 与本地一致，且远端变更 SHALL NOT 反向传播到本地

### Requirement: 同步范围采用显式白名单
系统 SHALL 仅同步 `src/`、`configs/`、根目录 `*.sh`、`*.ps1`、`pyproject.toml` 和 `uv.lock`。

#### Scenario: 同步根目录文件
- **WHEN** 根目录 session 扫描 GRID 工作区
- **THEN** 仅根目录 `*.sh`、`*.ps1`、`pyproject.toml` 和 `uv.lock` SHALL 被纳入同步

#### Scenario: 同步代码目录
- **WHEN** 目录 session 扫描 GRID 工作区
- **THEN** `src/` 和 `configs/` SHALL 通过相互独立的窄范围端点同步
- **AND** 根目录脚本 SHALL 由根目录白名单 session 同步

### Requirement: 远端重资产不受同步影响
系统 MUST 将远端 `.git/`、数据集、预训练模型、虚拟环境、日志、W&B 文件、实验产物和其他非白名单内容置于同步边界之外。

#### Scenario: 远端存在额外重资产
- **WHEN** 同步 session 创建、扫描、刷新或终止
- **THEN** 远端重资产 SHALL NOT 被扫描、下载、覆盖或删除

### Requirement: 受管目录保持完整副本
系统 SHALL 将受管目录中的所有远端额外内容视为非可信内容，包括运行时缓存和本地已删除目录中的残留文件。

#### Scenario: 远端存在运行时缓存或旧目录
- **WHEN** 远端在受管目录内生成 `__pycache__`、`*.pyc` 或保留本地已删除的旧代码目录
- **THEN** 下一轮同步 SHALL 清理这些额外内容，且任何内容 SHALL NOT 回传到本地

### Requirement: 首次同步具有显式传输门禁
命令行入口 SHALL 将 session 创建与首次文件传输拆分为两个显式操作。

#### Scenario: 用户启动同步项目
- **WHEN** 用户执行 `start`
- **THEN** 系统 SHALL 以暂停状态创建所有 session，不得开始项目文件传输

#### Scenario: 用户确认配置后恢复
- **WHEN** 用户执行 `resume`
- **THEN** 系统 SHALL 恢复项目 session 并允许本地受管代码同步到远端

### Requirement: 同步可通过统一命令行管理
系统 SHALL 提供仓库内 PowerShell 入口管理 Mutagen project，并 SHALL 忽略用户全局 Mutagen 配置。

#### Scenario: 用户管理同步生命周期
- **WHEN** 用户请求启动、查看、刷新、暂停、恢复、监控或终止同步
- **THEN** 入口 SHALL 将操作映射到固定的 Mutagen project 或具名 session 命令

#### Scenario: 前置条件缺失
- **WHEN** Mutagen、project 文件或必要的本地受管目录缺失
- **THEN** 入口 SHALL 在创建或修改任何 session 前以非零状态退出并给出明确错误

### Requirement: 验证不依赖真实远端
项目 SHALL 能通过轻量测试验证同步范围、方向、端点和命令映射，而无需连接 SSH 或启动 Mutagen daemon。

#### Scenario: 执行单元测试
- **WHEN** 测试环境没有可用的 `node1` 或 Mutagen daemon
- **THEN** 同步配置和命令行契约测试 SHALL 仍可完成且不得产生远端副作用

### Requirement: Agent 遵循同步完成门禁
项目 `AGENTS.md` SHALL 记录代码与重资产的可信源、受管范围、首次启动顺序和远端实验前的同步完成条件。

#### Scenario: Agent 准备远端实验
- **WHEN** agent 需要把本地改动交给远端实验使用
- **THEN** agent SHALL 通过管理脚本完成 flush，并在三个 session 均为 Watching 且无 conflict 后才报告同步完成
