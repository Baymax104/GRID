## 1. 模型

- [x] 1.1 标识 v0，提取基础loss复用边界并保持旧checkpoint兼容。
- [x] 1.2 实现v1当前候选表示、相关性head、联合loss和hybrid排序/trace。
- [x] 1.3 实现v1 checkpoint和显式v0 weights-only初始化。

## 2. 入口与数据

- [x] 2.1 提供固定selection/audit的共享数据组件，保留旧数据算法兼容。
- [x] 2.2 添加两版配置与脚本，支持dry-run/notes/override和DDP。

## 3. 验证与交付

- [x] 3.1 验证梯度、完整SID、标签独立、分片等价、恢复和数据/脚本/Hydra契约。
- [x] 3.2 OpenSpec strict、Mutagen flush/status及远端文件核验。
- [x] 3.3 更新版本说明、研究状态与可复制运行命令，报告未运行的效果与成本边界。
- [x] 3.4 修复四个脚本CRLF，补充原始字节回归与node1 Linux Bash启动参数核验，重新同步。
