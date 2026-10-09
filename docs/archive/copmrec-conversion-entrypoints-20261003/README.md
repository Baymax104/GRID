# CoPMRec 收益转化尝试启动文件归档

保存于 2026-10-03，包含退役前的 10 个根脚本、9 个 experiment 配置和 4 份启动契约测试。清理范围与验证结果见 [清理记录](../../copmrec-conversion-entrypoint-retirement.md)。

- `retired-files.zip`：原始文件字节，ZIP 路径保持仓库相对路径。
- `manifest.json`：每个文件的路径、大小、SHA256、ZIP SHA256，以及 17 份保留文件的清理前指纹。

这些入口已经退出标准目录。归档用于追溯，不能将其中的旧命令视为当前可运行命令。需要查看时在隔离目录读取，不自动解压回根目录或 `configs/experiment/`。
