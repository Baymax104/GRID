# CoPMRec v5.3：共享query的辅助无目录残差视图CE

## 授权、当前状态与累计成本

用户已批准唯一新增固定验证：1次推荐模型随机初始化、全部模块共同连续50000更新训练，随后1次own-best单卡完整Validation，新Testing0。正式训练[hqw189d2](https://wandb.ai/baymaxam/GRID/runs/hqw189d2)已正常exit0完成，物理GPU3、6→进程内[0,1]。主审计和独立复核通过：实际连续50000更新、100次raw Validation，own-best及完整保存状态为45000，完整预算由训练历史和终止日志另行证明。唯一完整Validation [6u6g62hk](https://wandb.ai/baymaxam/GRID/runs/6u6g62hk)已在物理GPU6→local[0]正常exit0完成；175文件／22363用户原始输出、附加独立复核与固定四组分析均通过。旧阶段3train／150000更新、3完整Validation／attempt4（其中预测前失败1次）／新Test0已经封口，旧账本按真实原字节归档；新增授权不是重新开始计数。

[新增额度登记](evidence/copmrec-unified-native-view-ce-50k-20261006/stage-registration.json)固定累计上限4train／200000更新、4完整Validation。登记时实际仍为3train／150000已完成、3完整Val、Test0，新增预留1train／50000及1完整Val。不分配seed43 pair或Testing，不扫描loss权重、temperature、alpha、checkpoint或seed。模型本身的固定预算仍与native42相同为50k／global256，比较不声称HPO成本、FLOPs或GPU时长相同。

整体双8目标、same-seed配对、实际原始输出审计及复现要求保持。v5.3相对native R10+7.0572%、N10+13.8794%，两绝对差CI下界均为正，但R10未达+8%。对v5.2仅+2.2918%／+0.8063%，两CI跨0；依既定分支停止该固定辅助CE、保留v5.2主方案，并保留v5.3相对native的明确部分收益证据。[本次完整结题](copmrec-v5-3-stage-decision-20261006.md)与[原阶段五项结题](copmrec-v5-2-stage-decision-20261006.md)。

## 问题、固定方法与决定

具体依据、额外计算量、初始content监督加倍和归因边界均按已审阅[固定proposal](copmrec-v5-3-native-view-ce-proposal-20261006.md)执行。剩余失败以seen商品竞争为主，cold占位已基本消除；native与v5.2仍有925新增／824丢失的命中互补。该事实支持一次固定监督干预，不证明残差目录或共享query是损失根因。

v5.3保持v5.2的联合目录训练和部署logits，仅训练增加权重1.0的无目录残差视图CE。同一shared query、同一次带dropout的目录projection，联合目录和无目录残差视图各计算一个matmul。无目录残差不表示history／query独立：history仍带共享协同残差，辅助CE会通过历史路径更新相关参数。没有teacher、外部推荐CP、分阶段optimizer重启或pool，不新增参数和部署视图。

原SID CE／joint content CE／legal-prefix mixture NLL、learned alpha、seen residual与cold0、完整目录支持集都保持；辅助CE也以全部目录为分母、仅允许seen训练目标。实际CPU预检已确认11031809个trainable参数、165个state tensors和两组空optimizer起点；实际checkpoint审计已证实154份optimizer moments及scheduler保存至45000。

训练配方保持DDP2每卡128／global256、FP32、AdamW WD0.035、主干peak0.0003／residual0.002、warm2500／cosine50000／min0／clip1，实际SID4层及输入Artifact不变；推理为单物理GPU映射local[0]。

主门禁对native42保持R10≥0.10470151589679383、N10≥0.0583728104417629，两paired绝对CI下界为正；冻结v5.2完整输出作描述性增量对照。通过后只称当前seed42合格，复现／Testing仍缺。若未通过但有清楚的部分推荐增益，按既有要求保留对应证据并披露其他指标的取舍；微小或CI跨0的增量只称不确定。若相对冻结v5.2没有明确推荐价值，停止这个固定辅助CE并保留v5.2，不自动换权重、第二query、续训或新预算。单seed开发集复用、pointwise CI与native42历史source缺口继续保留。

## 准备与实际证据

核心类、薄配置／根脚本、训练阶段辅助loss日志及严格21keys checkpoint契约已完成；74个核心与40个配置／脚本检查通过，独立静态review通过。新增`native_view_ce` exact8支持字典：固定protocol／weight1.0／training_only true／all_catalog／seen训练目标要求／catalog_residual false／shared_query和shared_projection true。

此前402 runtime字节保持；新增6个运行文件，真实408文件source SHA256为`d6c5000509d968caabf5b7d46b5c5be3889c0c9b5e857d60fe5cb1d4cf3b22a0`。官方Mutagen flush／status实际成功、三个session均Watching且无conflict；实际CPU预检没有forward／Trainer／CUDA，双卡smoke实际exit0、两个local rank、一步正常停止且W&B0。[实际实施凭据](evidence/copmrec-unified-native-view-ce-50k-20261006/implementation-verification.json)证明准备通过；训练及推理主审计已核真实source archive和408文件字节，不回填native42历史源码缺口。

当前累计training started4／completed verified4，committed与completed verified均为200000，完整Val started4／completed4、启动attempt5（此前预测前失败1次），Test0；新增训练及Validation均已消费唯一handle，新增额度用完。[实际账本](evidence/copmrec-unified-native-view-ce-50k-20261006/cumulative-budget-latest.json)与[实际终态登记](evidence/copmrec-unified-native-view-ce-50k-20261006/progress-validation-complete-receipt.json)。
