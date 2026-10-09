## Context

现有 v3 使用 root_max_mass processor，训练版本和 checkpoint 契约均为 v3；HF beam 使用 renormalize_logits=false。旧 trace v1 将 processor 输出视为条件 log 概率，不能直接容纳路径先验补正。

## Goals / Non-Goals

目标：完全保留首层分数与 frontier，复用 v3 checkpoint，用一次 root 包络补正检验跨 root 后续竞争问题，输出可核验的条件概率与搜索分数。

不包含：训练、新 head/loss、参数扫描、beam 扩容或完整实验自动启动。

## Decisions

新增 RootEnvelopeProcessor 派生现有概率 processor，仅额外聚合根层 Mass 表。根调用缓存每用户 rX、rM、rE=maximum(rX,rM)-logsumexp(maximum(rX,rM))；第二层按 beam 所属用户/root 读取 delta=rE-rX，对所有合法孩子加同一常数，其余层直接使用 v3 输出。每次 retrieve 创建独立 processor，禁止未经过根调用的后续调用，非法前缀保持 -inf。

新增 RootEnvelopeCoPMRec 派生 RootMaxMassCoPMRec，checkpoint_training_version 保持 v3，decode_version 为 v3.1。严格继承 v3 加载契约，不声称训练过 v3.1；训练接口明确拒绝。加入默认不变的 trace 工厂与 schema 钩子供新版本覆盖。

新 trace schema liger_paths_v2_root_envelope 使用专用观察器和 validator，原有概率字段保持局部条件值，新增 root admission/Mass/path prior/delta、目标搜索增量和累计搜索分数。只记录当前真实 frontier 可到达父节点；不可达项为 NaN。累计搜索分数使用真实 processor 输出沿目标祖先累加，第二层仅含一次 delta。v1 writer/validator 保持不变，新 callback 继承旧装配但切换 validator。

## Risks / Trade-offs

其他 root 提升可能挤掉 v3 原有命中，当前不保证效果；完整 Testing 必须同时报告恢复、损失和指标。归一化常数对同用户共同，仍保留以定义合法 root prior。修正步增量可能为正，独立字段允许此值，禁止伪装为条件概率。首层 HF 各初始 beam 分数必须一致，缓存按用户而非动态 beam 次序存储。

## Migration Plan

独立 model/config/script；现有 v0/v3 不变。局部 CPU/HF、配置、脚本验证后通过 Mutagen flush 并核验 node1 字节与 checkpoint；交给用户手动单卡运行。退回原 copmrec_v3_inference.sh 即使用原 decoder。

## Open Questions

方案能否减少后层遗漏并改善全体最终指标仍待用户完成唯一一次推理。Testing 已用于开发分析，结论保持开发证据范围。

## 固定推理 alpha 的追加对照

用户完成 zznw291t 后保留 v3.1，并授权固定 alpha=0.813 看最终效果。RootEnvelopeCoPMRec 接受现有 `inference_mixture_alpha` 字段，先验证有限且位于[0,1]，初始化父类时仍传空覆盖，随后保存为推理选项。checkpoint gate 参数不修改、不增加 state_dict 参数；v3 父类仍禁止其自己的推理覆盖，训练契约保持。

RootEnvelopeProcessor 在存在覆盖时使用同一常数混合首层 Max、后续 Mass 和根 Mass 先验；归一化 root 包络及第二层一次补正仍按原公式计算。null 时沿用原 gate 路径。首层规则保持，但权重改变可能改变 root 集合，不能要求与 zznw291t 首层 frontier 相同。同一 checkpoint 的 dense logits、排名和全部参数必须相同。

candidate/path metadata 分别记录实际 content_mixture_alpha、mixture_alpha_source=fixed_inference、checkpoint_mixture_alpha 与固定 mechanism 标记。继续使用既有 trace schema 和 writer。根脚本不新增参数，只透传 `model.root.inference_mixture_alpha=0.813`。单卡 full Testing 一次、零训练；与 zznw291t 对照，主要评价 NDCG10，联合看 Recall10、81个高内容遗漏和81个现有超dense增量命中的损益。正向支持此固定解码设置，负向回退原学习值，不确定不自动扫描；完整运行仍由用户手动开始。
