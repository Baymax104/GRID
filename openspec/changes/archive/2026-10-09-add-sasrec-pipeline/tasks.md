## 1. 装配和命令

- [x] 1.1 添加独立 data/model/trainer/logger/callback 与薄 experiment 配置。
- [x] 1.2 添加根目录脚本及参数验证。
- [x] 1.3 编写中文算法忠实度、协议差异及手动命令说明。

## 2. 验证与交付

- [x] 2.1 通过Hydra compose、脚本语法与quoting/空值/错误/override测试。
- [x] 2.2 通过统一入口本地dry-run与5 step，恢复推理并核验checkpoint和bundle。首次5 step退出编码失败；UTF-8下恢复step5正常退出且未增加步数，详情见基线说明。
- [x] 2.3 完成聚焦回归、Ruff、四模块strict与全目标审计，更新验证记录。
