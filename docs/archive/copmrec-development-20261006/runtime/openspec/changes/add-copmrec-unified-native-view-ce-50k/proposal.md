## Why

v5.2已完成连续50k及完整Validation，相对同policy native42 R10+4.658672%、N10+12.968595%，双paired CI正，保留部分收益但未达原双8门槛。cold错误占位仅44槽／42用户，不足解释824个native命中丢失；现有seen命中互补支持一次共享query的目录视图监督鉴别，不证明残差目录或query是根因。原3train／150k及3完整Val已结题，用户随后明确授权本次新增有界验证。

## What Changes

- 新增v5.2派生v5.3，只在训练增加固定权重1的无目录残差视图全目录CE；原三loss、联合目录部署评分与随机起点50k协议保持。
- 固定`native_view_ce` exact8契约，模型／checkpoint／顶层配置／共享writer metadata一致；新增薄配置、根脚本与聚焦测试，旧402运行文件字节保持。
- 显式追加1train／50000更新＋1完整单卡Validation，新增Test0／扫描0。旧成本保留，累计达到4train／200k及4完整模型Val；本次实现不等同正式运行或效果。

## Capabilities

### New Capabilities

- `copmrec-unified-native-view-ce-50k`：共享query无目录残差视图CE的连续scratch50k与单checkpoint评价契约。

### Modified Capabilities

无。

## Impact

新增recommendation类／聚焦测试、model／experiment配置、根训练／推理脚本及配置测试；不修改既有402文件或baseline默认行为，无新依赖／参数／推理视图。root单独登记预算并负责同步、实际预检、正式运行及只读审计。

依据：[v5.2结题](../../../../docs/copmrec-v5-2-stage-decision-20261006.md)、[授权前方案快照](../../../../docs/copmrec-v5-3-native-view-ce-proposal-20261006.md)。快照中的“尚未授权”是其形成时事实，本change记录之后的用户明确开始授权。
