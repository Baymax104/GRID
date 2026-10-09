# run-source-snapshot Specification

## Purpose
保存正式运行使用的实际源码字节、运行环境及可信本地来源，保证 W&B code Artifact 可独立追溯。
## Requirements
### Requirement: Runtime source capture
系统 SHALL 在正式统一入口执行任务前归档实际源码、配置、根目录脚本和依赖清单；SHA256 manifest MUST 与归档字节一致，并随档记录关键运行版本。

#### Scenario: Dirty source survives capture
- **WHEN** 源码存在未提交修改或新增 Python 文件
- **THEN** 归档包含这些实际内容及对应指纹，不依赖 git tracked 文件列表

#### Scenario: Asset exclusion
- **WHEN** 仓库包含数据、checkpoint、日志、缓存、虚拟环境或 .env
- **THEN** 快照只包含白名单文件，不跟随符号链接

### Requirement: Trusted local origin
本地同步入口 SHALL 生成 commit/dirty 与文件指纹来源记录；运行端 MUST 不查询远端 Git，只有指纹匹配时才将来源标记为 verified。

#### Scenario: Changed remote file
- **WHEN** 运行源码指纹与本地记录不同
- **THEN** 来源标记 mismatch，归档仍记录实际内容，不声称匹配本地 commit

### Requirement: Publication before execution
启用 W&B 的正式运行 SHALL 在任务执行前上传真实文件为 code Artifact，并在 resolved config 中记录快照指纹；快照或发布失败 MUST 传播异常。

#### Scenario: W&B source artifact
- **WHEN** 配置启用 W&B logger
- **THEN** code Artifact 保存归档、manifest 和环境信息，可脱离运行机器下载

### Requirement: Dry run and distributed behavior
系统 MUST 在 dry-run 跳过快照写入/发布，在多进程模式仅由全局 rank0 生成和发布。

#### Scenario: Nonzero rank or dry run
- **WHEN** 当前为 dry-run 或全局 rank 非零
- **THEN** 不创建源码快照，不上传 code Artifact
