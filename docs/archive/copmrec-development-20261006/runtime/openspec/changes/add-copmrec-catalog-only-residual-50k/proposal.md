## Why

v5.2已确认相对native42的部分推荐收益，但seen命中新增925／丢失824；v5.3辅助CE对v5.2增量未确认，已停止。四个scratch候选都共享history和catalog residual，尚未检验history侧显式residual是否值得保留。已审阅的固定proposal限定在这一结构选择，不预判query为损失根因。

用户长期目标允许推进本地可逆实现；proposal形成时4train／200k／4Val额度已用完，新额度待答复。2026-10-06T11:01:32.778Z用户明确“批准运行 v5.4”，只批准既定1train50k＋1单卡完整Val，累计5／250k／5，Test／43／扫描均0。代码准备、额度登记和实际运行分别记录，不能由此变更主方案或宣称效果。

## What Changes

- 从v5.2派生v5.4，仅取消history residual，保留catalog residual、原三unit loss、learned alpha和单目录部署。
- 薄类覆盖history所用hook并显式调用原catalog residual hook，不修改旧父模块，不增加参数／teacher／第二query／辅助CE。
- 严格新增placement契约，准确表达history=false、catalog=true，两处sharing和版本一致，拒绝旧语义checkpoint。
- 新薄Hydra配置、根训练／推理脚本与聚焦内存测试，统一入口、notes、dry-run、额外override及writer身份保持。
- 本地准备可先完成；实际CPU／DDP2 smoke、正式50k及单卡完整Val须在明确新增额度之后。双8及复现目标不降低。

## Impact

新增推荐模型、配置、脚本、测试与研究记录；旧408 runtime字节保持。没有新依赖或旧checkpoint兼容承诺。现有v5.2主方案和v5.3部分证据、停止决定、闭合预算不变。固定方案详见`docs/copmrec-v5-4-catalog-only-residual-proposal-20261006.md`。
