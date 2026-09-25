## 1. 同步配置

- [x] 1.1 创建仓库级 `mutagen.yml`，声明四个窄范围 `one-way-replica` session
- [x] 1.2 为根目录 session 配置严格 allowlist，仅同步启动脚本和依赖清单

## 2. 命令行管理

- [x] 2.1 创建根目录 `mutagen_sync.ps1`，实现前置条件检查和仓库根目录解析
- [x] 2.2 实现 `start`、`status`、`flush`、`pause`、`resume`、`monitor`、`stop` 操作，并确保首次启动保持暂停

## 3. 自动化验证

- [x] 3.1 添加同步配置静态测试，覆盖方向、端点、白名单和重资产隔离
- [x] 3.2 添加 PowerShell 命令入口契约测试，并执行聚焦测试与脚本语法检查

## 4. 集成验证

- [x] 4.1 运行 `openspec validate add-mutagen-code-sync --strict` 并检查最终工作树
- [x] 4.2 以暂停状态创建 Mutagen project，核对 session 后恢复并刷新，验证远端受管代码与本地一致且重资产仍存在

## 5. Agent 工作流

- [x] 5.1 将可信源、同步边界、生命周期命令和实验前完成门禁写入项目 `AGENTS.md`，并增加静态契约测试
