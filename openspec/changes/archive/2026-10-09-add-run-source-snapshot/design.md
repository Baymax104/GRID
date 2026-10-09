## Context

pipeline_launcher 在装配完成后记录配置，随后进入训练/预测/分析；目前没有源码留档。源码来自本地 dirty 工作树，通过三个 Mutagen session 同步到 node1；远端 Git 不可信。

## Goals / Non-Goals

目标：完整保留实际运行的核心源码、配置、启动脚本及依赖版本，准确识别本地来源。
范围外：历史 run 补录、实验重跑、算法调整、远端 Git 修复。

## Decisions

- 快照生成归 utils，发布归 common/writers，launcher 串联二者。正式运行在组件装配前读取文件，所有 hash 和 tar 内容来自同一批字节；发布和配置记录发生于任务执行前。
- 固定白名单：src 的 Python 文件、configs 的 YAML 文件、根目录 sh/ps1、pyproject.toml、uv.lock；不读取 data、logs、虚拟环境、checkpoint、缓存或 .env。符号链接拒绝，防止越界归档。
- 归档写入 paths.output_dir/metadata/source_snapshot，包含 source.tar.gz、manifest.json、runtime.json。启用 W&B 时用 add_file 上传真实内容为 code Artifact，不使用本地引用；manifest hash 写入 resolved config。
- 本地 start/resume/flush 前调用纯标准库工具生成 src/.source_origin.json，记录本地 commit/dirty 和源码清单指纹。运行端不执行 Git；来源缺失或指纹不匹配时标记 unavailable/mismatch，不将其当作实际版本。
- 只在 rank0 执行；dry-run 不产生快照或发布。默认启用，显式 source_snapshot.enabled=false 可关闭。快照/发布失败传播异常，避免正式运行悄悄丢失留档。

## Risks / Trade-offs

- [同步后文件变化] → 比较运行端实际 manifest 与本地来源指纹，拒绝宣称匹配；归档记录实际字节。
- [新增启动耗时] → 仅扫描核心代码且不复制重资产，不新增包。
- [多卡重复发布] → 初始化前检查全局 RANK，并兼容已初始化的 distributed rank。
- [正在运行的旧 run] → 只影响下次启动，不为历史 run 补造源码证据。
