# W&B 历史机制 run 清理核对（2026-10-08）

用户授权：确认历史机制 run 可以删除后直接删除。本次实时核对结果：旧 run 已在此前清理中删除，没有新增删除对象。

## 当前 W&B 状态

- Entity：baymaxam；可访问项目清单只有 GRID。
- GRID 的 SDK 全量盘点共99个现存 run，无分页遗漏。现存 group 为正式主结果/基线与必要上游；没有历史机制 group，也没有 run_mode=analysis 的旧诊断。
- 针对旧 Legal/Max/Mass 的9个精确 run ID，独立 W&B GraphQL 查询返回0条，hasNextPage=false。
- 此前 docs/archive/copmrec-development-20261006/wandb-delete-20261007/deletion-receipt.json 与 verification.json 已记录这些 run 的删除及不存在核对。
- 本次新增删除 run 数：0；新增删除 Artifact 数：0。保留全部现存正式主结果、基线及必要上游。

| 数据集/seed | 旧Legal | 旧Max | 旧Mass | 当前现存数 |
| -- | -- | -- | -- | -- |
| Beauty/42 | jpqimr2j | sgmu9l3l | 6dspa7e3 | 0 |
| Sports/42 | sd90tlbj | z8ymlal9 | siiokyhl | 0 |
| Toys/42 | bzgzloh5 | 1raiqpvi | valgk6yg | 0 |

删除资格依据当前 research-state.yaml：开发实验只供文档追溯，不复用为正式证据；清理不能改变历史事实。旧配置、指标与成本回执继续保留在本地归档。

本次只盘点和核对，没有创建实验、修改Linear计划、删除项目或删除本地归档。全量身份清单见 inventory.json。
