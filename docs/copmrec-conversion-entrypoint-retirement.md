# CoPMRec 旧收益转化尝试入口清理

日期：2026-10-03。授权来源：用户要求清理训练后冻结的收益转化尝试入口，保留其做法和结果作为重新设计的参考。

## 清理结果

已退出标准启动目录的文件共 23 份：10 个根目录 shell 脚本、9 个 Hydra experiment 配置及 4 份仅服务这些入口的启动测试。所有原始字节在删除前完成 ZIP CRC、大小和逐文件 SHA256 校验。

归档为 [retired-files.zip](archive/copmrec-conversion-entrypoints-20261003/retired-files.zip)，完整路径和哈希见 [manifest.json](archive/copmrec-conversion-entrypoints-20261003/manifest.json)。归档存放于本地文档目录，未加入 Mutagen 的代码同步范围。

| 尝试 | 已退役的 experiment 名称 |
|---|---|
| decoder 单独微调、单/双分支保护、高位 cap、边界辅助 | `copmrec_decoder_train`、`copmrec_decoder_audit` |
| 独立小型 pairwise ranker | `copmrec_ranker_train`、`copmrec_ranker_inference` |
| 中途复制冻结 teacher 的 scratch 方案 | `copmrec_scratch_train`、`copmrec_scratch_train_ddp2`、`copmrec_scratch_audit` |
| 冻结特征缓存、mixed 路径终排 | `liger_joint_ranking_cache`、`liger_joint_ranking_inference` |

基础 CoPMRec 的 `liger_joint_train` / `liger_joint_inference` 和 LIGER 的 `liger_train` / `liger_inference` 保留。基础脚本、配置及相关算法共 17 份文件与清理前 SHA256 一致。

旧算法、component 配置、writer 和算法单元测试继续保留，用于理解既有 checkpoint、trace 和技术行为。手工装配 component 仍然可能；本次退出的是标准启动入口，不把参考实现视为当前推荐路线。数据、checkpoint、日志和 W&B 产物保持原有保管规则。

## 验证

- 聚焦 pytest：108 passed，覆盖归档完整性、退役 experiment 无法 compose、基础配置 compose、基础脚本语法及参数、保留的相关算法。
- `openspec validate retire-copmrec-conversion-entrypoints --strict` 通过；新退役测试的 Ruff 检查通过。
- 使用项目入口完成 Mutagen `flush`，随后三个会话均 connected / `Watching for changes`，无 conflict。
- SSH 只读检查 node1：19 个受管入口全部不存在，17 个保留参考文件全部存在。远端 Git 未用于判断同步结果。
- 没有启动新训练或推理，没有操作已启动的运行进程。

## 研究定位

BMX-116 的基础版本结果继续有效，但相对 LIGER dense 的最终收益尚未明确建立。清理入口不否定已完成实验，也不代表完整方法已经完成。

旧尝试做法、效果、限制以及新组件设计见 [收益转化组件重设计](../../research/docs/2026-10-03-copmrec-conversion-redesign.md)。新组件目前为设计，尚未接入生产模型；完整实验由用户手动开始，旧阶段预算不自动转移或重置。
