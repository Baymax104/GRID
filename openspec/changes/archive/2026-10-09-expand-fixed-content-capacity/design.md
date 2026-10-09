## Context

用户明确授权一次扩容。瓶颈尚未定位；本次测试更丰富PCA表示与更宽网络的组合效应，不单独归因于PCA信息或容量。原2k/10k适配器失败保持，未证明内容表示家族无效。

## Goals / Non-Goals

一个新训练和一个配对评价；不重训128基线、不接回CGBS、不加层、不改变FFN或预算。

## Decisions

PCA256直接固定，adapter模块/参数完全不存在。2层Transformer、4heads、FF512、dropout0.1；隐藏/查询256。固定10k、batch32、seed42、同校准排除与数据流，纯item CE。基础bank不可训练。新protocol与目录指纹隔离旧实验。

复用128固定dvahqxw2，比较数据metadata/流/步数一致；不要求不同维度初始化SHA或bank相同。显式检查各自维度、adapter存在性、SID一致及来源。共512既有开发用户；报告原始NLL/Recall/NDCG、rank和配对差异，不视为独立确认。

## Risks / Trade-offs

支持依据仅有128固定模型随预算改善；未证明容量不足，增加维度也可能更易拟合有限缓存。语义adaptation和Transformer已有UniSRec/LIGER先例，本次工程扩容不提供新颖性。以配对NLL改善CI下界>0且Recall不降作为继续审查资格；失败停止此配置，不自动扩预算。实现契约失败与有效负面区分。模型宽度及输入维度同时改变，不能宣称单变量容量因果效应。
