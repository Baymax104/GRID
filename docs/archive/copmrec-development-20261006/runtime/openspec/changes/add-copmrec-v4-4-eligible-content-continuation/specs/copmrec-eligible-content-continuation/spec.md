## ADDED Requirements

### Requirement: 只修改训练content CE历史支持

系统 SHALL 在真实v4.4的treated中仅将训练content CE原cold=-100之后的有效完整SID历史行设为-inf；control SHALL 保留原content CE。SID CE和mixture NLL MUST 使用原定义及raw full-catalog logits，三loss权重各1、learned alpha不变；validation loss定义仍原样。

#### Scenario: 同一表示的三loss
- **WHEN** 相同权重和batch计算treated/control的SID、content、mixture项
- **THEN** 原encoder/query/raw logits只计算一次，treated只替换content项，SID/mix同权重下逐值相同，不能将mask后的logits送入mixture processor

#### Scenario: control与旧loss匹配
- **WHEN** exclude_history_from_dense_ce=false
- **THEN** 三loss与原v4.1定义及梯度相同，未增加其他训练机制

### Requirement: 资格mask保留CE梯度与有效target

系统 MUST 沿用v4.3最多20有效完整SID、二值/rightpadding/精确catalog lookup/inactive忽略/去重语义；训练与Validation MUST 验证target不在有效causal history。被排除历史行的content CE直接梯度 SHALL 为0，未排除行 SHALL 保留正常autograd，loss/梯度有限；不能直接使用no_grad推理helper的分数返回值作为训练loss。

#### Scenario: 历史竞争行被排除
- **WHEN** treated训练batch有合法完整历史和未见于历史的seen目标
- **THEN** 只有history对应CE logits为-inf，其他cold仍原-100，目标未被排除，eligible行继续有CE梯度

#### Scenario: label重复或无效SID
- **WHEN** target出现在有效history，或history包含partial/unknown SID或非法attention
- **THEN** 准备或batch明确失败，不静默删样本、不以label修改mask，也不将目标放回资格集合

### Requirement: 严格同源weights-only初始化与真实continuation链

两臂 SHALL 使用同l3zyr91b原v4.1 best6000及SHA4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9，通过公共pretrained_v4_1_checkpoint和原native v4.1 hook/strict state验证；MUST 保留原v0/v4链、新记录真实v4.1 continuation来源，不修改原checkpoint版本，不恢复optimizer/scheduler/global_step。顶层ckpt_path MUST null，旧v0/v4初始化字段null、bias0冻结。

#### Scenario: 两臂从完全相同权重开始
- **WHEN** 公共loader解析固定warm URI和bytes
- **THEN** 两臂全部初始state与原l3相同，bias全0，catalog和来源契约一致，native辅助构造不扰动相同RNG起点

#### Scenario: normal v4.4 restore
- **WHEN** 恢复新真实checkpoint
- **THEN** 严查真实version/step、exclude flag、copmrec_continuation_pretraining和exact9 copmrec_content_eligibility，以及原父契约/strict state；不桥接新文件成旧v4.1/6000

### Requirement: 固定匹配训练配置与有界终态

两臂 MUST 除exclude_history_from_dense_ce外相同：FP32/seed42、GPU2,4→local0,1、batch128/card/global256、AdamW主干LR.0001/item.002/WD.035、bias冻结、residual_multiplier20、warmup300、scheduler_steps显式6000、max_steps2000、val每1000；不缩短cosine horizon为2000，不扫描参数。

#### Scenario: 完整预算与checkpoint选择
- **WHEN** 每臂完成2000个optimizer updates
- **THEN** 核验1000/2000两次history-excluded dense Validation N10及实际best Artifact，选其自身best1000或2000并记录固定2000标量，不用Testing选择

### Requirement: 固定history-pool部署真实新continued成员

每臂 SHALL 只使用固定nj9elah1 v4 best6000 source与自身selected新v4.4 continued checkpoint，各独立query/catalog full logits等权0.5/0.5，再应用原exact9 history_eligibility_contract并stable catalog-row Top10。MUST 验证新continued真实URI/SHA/step/flag及完整来源链，旧pool默认检查不放宽，pool顶层ckpt_path=null、单卡NPROC1、标准keys/predictions writer、无wrapper checkpoint。

#### Scenario: 正常部署新best
- **WHEN** selected Artifact真实step为1000或2000且所有来源/目录契约一致
- **THEN** 以真实新身份正常strict恢复，cold可选，原bias0与cold residual0保留，不预填旧l3 SHA作为新best

### Requirement: 统一入口与完整可复算证据

系统 SHALL 使用src.main/Hydra、公共loader/lineage、实际source archive及commonwriter；根训练/推理脚本 MUST 支持quoted URI、notes两写法、显式dry-run、empty/错误参数拒绝与用户extraoverride。正式preflight MUST 冻结训练/资格/来源条件、Mutagen flush三Watching、实际动态全runtime字节及production batch/DDP smoke。

#### Scenario: 两个完整pool Validation
- **WHEN** 两真实selected checkpoint均ready
- **THEN** 各仅一次完整单卡Evaluation，独立175raw/22363用户末商品label、SID合法唯一/零history重叠、input/catalog/checkpoint/source/actual metadata核验，复算fair LIGER wdms8w77、旧v4.3 iy3o3z3q、treated-control的R/N/pairedCI/newlostshared/固定组warmcold，不新增baseline Validation

### Requirement: 显式追加而非重置预算

root登记的新阶段 MUST 将旧5训练/30000封存并显式追加2×2000，累计limit7训练/34000；新pool完整Validation最多2，条件Testing最多2，全线程Testing仍3且已用1。任何失败或不确定结果 MUST 不自动重置额度、扫描或延长续训。

#### Scenario: 两臂不合格
- **WHEN** 新两臂均不满足整体晋级
- **THEN** 不Testing、不续训/换参数，结束本有界阶段，保留已有效v4.3历史层及符合独立保留标准的正向续训贡献

### Requirement: 自身保留与history CE增量归因分开

任一新pool对旧iy3o3z3q的R/N任一相对提升≥3%且另一无点退化时 SHALL 可保留开发贡献，CI/groups约束效果主张；treated-control MUST 独立评价mask增量，control本身可保留。不能以CE下降、频率/content关联或总体达门槛声称history CE因果收益。

#### Scenario: control有效而mask不确定
- **WHEN** control自身明显改善，treated-control CI跨0或负向
- **THEN** 可保留续训贡献但不确认mask有效，不复活旧bias/CF/mixed，不因此扫描参数

### Requirement: 预承诺唯一Validation部署选择与条件Testing

系统 SHALL 仅在raw/source审计通过且新pool R10≥fairwdms8w77的1.1倍、N10≥1.1倍、两paired差值CI下界>0、旧ValidationR≥.09877029021151008/N≥.0511173919307933的合格两臂中选择唯一部署。排序 MUST 按实际N10降序，精确同N按R10降序，再精确同R选treated；不按有利cohort替代整体。

#### Scenario: 唯一合格winner固定确认
- **WHEN** 存在按此固定规则选出的qualified winner
- **THEN** 至多进行同policy固定LIGER与该pool各一次Testing，全线程1→3；Testing不选择模型，接受须对新same-policy baseline双≥1.1、两pairedCI正且旧042139al R≥.07855386128873586（1757hits）/N≥.04002978116676896

#### Scenario: 有明显组件收益但整体未达
- **WHEN** 自身保留标准满足而整体R或N门槛未达
- **THEN** 保留有效组件和真实证据、整体goal仍active、不触发Testing或自动成本；未知原因不写成已确认支持集有害
