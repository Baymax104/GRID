# 正式运行源码留档

从统一 `src.main` 入口启动的正式训练、推理和分析默认在任务执行前留档，不需要为每个 experiment 另行配置。dry-run 不留档，多卡仅由全局 rank0 执行。

## 留档内容

运行端实际文件保存到 `${paths.output_dir}/metadata/source_snapshot/`：

- `source.tar.gz`：`src/**/*.py`、`configs/**/*.{yaml,yml}`、根目录 `*.sh`/`*.ps1`、`pyproject.toml`、`uv.lock`，包括未提交及新增文件。
- `manifest.json`：逐文件 SHA256/大小、整体源码指纹、归档指纹、来源核验状态。
- `runtime.json`：Python、平台、torch/lightning/W&B/Hydra 等版本及 PyTorch CUDA build；不收集认证信息或环境变量。

数据、日志、checkpoint、虚拟环境、缓存和 `.env` 不进入归档。归档只包含普通文件，不跟随符号链接。文件 hash 与归档取自同一批读取字节。

启用 W&B logger 时，运行开始前将三个文件通过 `add_file` 上传到 `grid-source-<run-id>`，类型为 `code`、role 为 `source_snapshot`。这是实际文件上传，不依赖 node1 本地文件引用。resolved config 中的 `source_snapshot_record` 保存源码/manifest指纹和本地目录，供后续 run 审计查询 `logged_artifacts()` 对齐。Artifact 名称避开 SDK 保留的 `source-` 前缀。

快照或发布失败会中止任务，避免正式运行丢失留档。提交上传后由 W&B SDK 在 run 生命周期中完成传输，审计时仍须检查 Artifact 为 COMMITTED。

## 本地来源

从本地仓库根目录执行 `./mutagen_sync.ps1 start`、`resume` 或 `flush` 时，会在实际同步前生成 `src/.source_origin.json`。该文件被 Git ignore，但通过 src session 单向同步，记录本地可信 commit、dirty 状态和源码指纹。

运行端不读取 Git。其实际源码与本地记录匹配时，manifest 的 `origin.status=verified`；缺失时为 `unavailable`，不一致时为 `mismatch`，损坏时为 `invalid`。后两种情况仍保存实际运行源码，但不将来源 commit 作为已核验版本。

完成修改后应执行 `./mutagen_sync.ps1 flush`，确认三个 session Watching for changes 且无 conflict，再手动启动正式实验。若之后又修改了受管文件，应再次 flush 刷新来源记录。

Git dirty 表示整个本地工作树存在变更；源码指纹只覆盖上述运行白名单，不代表论文文档或所有仓库文件的版本。

## 使用边界

显式设置 `source_snapshot.enabled=false` 可关闭留档，正式可复现运行应保持默认开启。已启动的旧 run 不会被补录；BMX-69 的缺失源码快照限制继续保留，不能将当前快照冒充历史运行源码。

## 2026-10-01 验证记录

聚焦测试85项通过，涵盖归档字节与hash、dirty/untracked、来源缺失/损坏/不匹配、rank/dry-run跳过、发布异常、launcher执行顺序、SASRec/TIGER/LIGER配置组合及原有入口/日志回归。Ruff、PowerShell语法与OpenSpec strict通过；同步脚本的来源生成先于flush、status不修改来源、生成失败阻止同步均经过隔离stub验证。

Mutagen flush成功，三个session均为Watching for changes且无conflict。node1真实源码快照包含226个白名单文件，归档约226kB；逐文件tar内容hash均与manifest一致，origin.status=verified，git_dirty=true，本地与实际运行源码整体SHA256为`180b336da2191b9b22e28e054130ba0555ec4b88fb06e287f72de7f96b0b0432`。

node1真实W&B SDK离线code Artifact发布与配置记录成功，未创建线上run、未运行训练/推理。验证目录为`node1:/data3/weizhenyu/projects/GRID/logs/source-snapshot-validation/node1-20260930-161415`（目录使用node1的UTC时间）。线上Artifact的COMMITTED状态留待下一个正式run审计确认。
