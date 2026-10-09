## Why

当前v4 best6000在完整Validation上相对LIGER dense提升6.92%/7.15%，但错误Top10中最高频20件商品仍占18.51%，两个固定用户子组各约18.52%、Top20重合19件。现有normalized cosine没有独立商品打分截距；检验该自由度能否改进稳定误排，不将曝光比例当作概率校准或缺少bias的因果证据。

原残差阶段3×6000步已完整结题：保留残差正收益，关闭mixed终排与LR序列；本变更是明确登记的新问题与一对匹配续训，不重置原预算。用户已授权本自主目标SSH训练/推理及新增模块，最终Testing两项≥10%目标保持。

## What Changes

- 新增v4.1派生模型：cosine/temperature后增加seen-only零初始化商品scalar bias，历史表示保留v4共享残差。
- 三项原损失保持；bias直接接受content CE和mixture NLL监督，SID CE没有对纯打分bias的直接梯度。
- 严格从同一v4 best6000做weights-only初始化，新增bias学习或固定0；新optimizer/scheduler，不恢复旧训练步数。
- 新增薄配置/根训练与单卡推理入口、CPU契约及旧模型等价回归。
- 新阶段最多2训练臂×6000步/两次完整单卡Validation，沿用本线程最多3次Testing确认；累计训练最多5臂/30000步。

## Capabilities

### New Capabilities

- `copmrec-item-score-bias`：商品截距共享目录loss/推理、cold mask、匹配zero-bias与严格warmstart/恢复。

### Modified Capabilities

无。既有v0/v4配置、状态与默认行为保持。

## Impact

新增recommendation模型、component/experiment配置、根脚本和聚焦测试；复用artifact loader、共享writer、统一src.main/Hydra与source snapshot，不引入依赖。新增12101个参数，cold33行始终不生效；dense部署明确不声称生成候选贡献。商品bias是成熟组件，不宣称首创，定位及完整五项门禁见对应研究记录。
