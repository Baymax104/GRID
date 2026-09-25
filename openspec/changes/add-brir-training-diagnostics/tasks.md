## 1. 证据与入口

- [x] 1.1 扩展 reference 返回及可选 audit v2，记录训练目标与全目录排序证据
- [x] 1.2 实现证据校验、有界四 arm 手动套件和执行说明

## 2. 验证与交付

- [x] 2.1 验证零更新一致性、候选支持差异、残差边界和旧行为兼容
- [x] 2.2 完成 Hydra compose、shell 参数验证与 Sports 匹配协议核对
- [x] 2.3 完成 Ruff、OpenSpec strict、Mutagen flush/status 并交付手动命令

验收（2026-09-16）：64项聚焦测试通过，Ruff与OpenSpec strict通过，git diff --check无空白错误；仅有现存依赖弃用及Git换行提示。Mutagen flush成功，四个session均Watching for changes，无conflict。完整GPU任务未启动，未提交或归档。手动命令与判读顺序见runbook.md。
