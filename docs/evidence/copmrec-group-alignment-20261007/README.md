# CoPMRec 主实验 group 格式统一

2026-10-07按用户要求，将CoPMRec分组与其他实验的paper_main_<method>_<dataset>统一。

| 数据集 | 训练与Testing共用group |
|---|---|
| Beauty | paper_main_copmrec_beauty |
| Sports | paper_main_copmrec_sports |
| Toys | paper_main_copmrec_toys |

- 9个Linear主结果issue的18条命令全部更新；标题、状态、其余正文与命令参数不变，未增加Validation。
- copmrec_train与copmrec_inference的Hydra默认group同步更新；root脚本透传方式保持，显式group override继续生效。
- 正式计划命令补充统一的显式group；版本定义、当前计划、研究状态、Run Registry与在线Linear文档同步。
- 版本、正式身份、seed与split仍由config/notes/tags记录；group不再包含v5.3或formal。历史快照保持原记录，不追改已发生的run。
- 验证：45项现有配置/脚本测试通过（含Hydra与Bash检查）、三数据集两入口group组合通过、OpenSpec strict通过、9个issue在线回读一致。源码/配置修改使用官方Mutagen入口同步node1。

本次未启动实验、未修改模型计算或训练预算、未创建Git提交。回执包括issues-before.json、issue-proposals.json、linear-update-receipt.json、linear-readback-verification.json与local-verification.json。
