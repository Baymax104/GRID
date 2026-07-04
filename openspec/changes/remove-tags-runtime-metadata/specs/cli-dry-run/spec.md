## MODIFIED Requirements

### Requirement: Dry run MAY preserve runtime metadata outputs
dry run 第一版可以保留 Hydra 输出目录和运行元信息写入，只要不写入业务结果。

#### Scenario: Dry run preserves operational logs
- **WHEN** dry run 运行
- **THEN** 允许生成 Hydra 输出目录与普通运行日志
- **AND** 允许保留 `config_tree.log`
- **AND** 不得要求保留已删除的 `tags.log`
