## Context

本提案延续 A=`token_content_init`，不是旧 CGBS 在线评分分支。当前 A 使用固定 PCA/L2 内容均值初始化共享 SID 输入表；输出 head 独立，合法 CE、beam10、20k steps、每500步 validation 保持原协议。实现及轻量验证已完成，完整实验未运行；交付记录见 [implementation-results.md](implementation-results.md)。

## Goals / Non-Goals

目标：隔离第二层固定内容交互作为持续输入先验的增量价值；从原 A 初始函数出发，控制额外参数和归因成本。

非目标：更改 tokenizer、SID、输出 head、辅助 loss、搜索预算、范数校准、第三层扩展、教师蒸馏或完整 prefix memory；本轮不确立论文贡献。

## Decisions

### 1. 物品等权的加性分解

固定目录内容 x_i、第一层码 p_i、第二层码 k_i。对全部合法目录物品等权拟合：

```text
min_{a,b} Σ_i ||x_i - a[p_i] - b[k_i]||²
μ[p,k] = mean(x_i | p_i=p,k_i=k)
r[p,k] = μ[p,k] - a[p] - b[k]
```

等价于对占用 pair 的均值作 n[p,k] 加权最小二乘。目录占用数不是训练交互频次。不得在不平衡目录上直接用 μ[p,k]−μ[p]−μ[k]+μ_global 替代拟合。

在 CPU float64 上固定顺序交替组均值：a←mean(x−b)，b←mean(x−a)。每轮在二部图各连通分量内将 a 的物品加权均值归零，同时反向平移 b；a/b 的 gauge 不唯一，但已观测边上的拟合值与交互残差唯一。按键排序保证稳定。以两侧分组残差均值的最大绝对值≤1e−10为停止条件，最大1000轮；未收敛或非有限则报错，不静默采用近似残差。

正式训练实现直接使用公共 loader 得到的相同目录构造该 buffer；研究中复用 C checkpoint 的固定目录仅是方便，不允许把 C checkpoint 新增为 A 改进的上游依赖。

### 2. 首个候选只学习一个标量

令第二层初始 A 表为 E_A(0)，对合法目录物品计算：

```text
q_A = sqrt(mean_i ||E_A(0)[k_i]||²)
q_r = sqrt(mean_i ||r[p_i,k_i]||²)
R[p,k] = q_A / q_r * r[p,k]
E'_2(p,k,t) = E_A[k,t] + tanh(g) * R[p,k]
g(0)=0
```

q_A、q_r及R一经构造即冻结并持久化。q_r≤1e−8或q_A≤1e−8时报错。是全局能量校准，不做逐行归一化，保留残差相对强度。正负门值均允许；标量与A参数使用相同Adam设置，不单独扫学习率。只增加一个可训练参数，encoder/decoder共用它。选择单标量以先判断固定内容方向是否可用，其失败不否定可学习适配器。

不使用 alpha=0 且 W=0 的双零乘积分支，该结构对两者梯度均为零。新结构初始化不消耗全局随机流；所有 A 参数须与相同 seed 的参考逐位一致。归零门的代数回退不代表训练后关闭门可还原未经分支训练的 A。

### 3. 固定输入对照

- `interaction`：上述R。
- `additive`：v[p,k]=a[p]+b[k]−μ_k，提供相对 A 原型的加性前缀信息；用同一全局RMS规则缩放，同样只学习一个标量。
- `shuffled`：使用局部seed42在每个第二层码内打乱pair残差行，随后按原n[p,k]权重重新移除父码/当前码加性成分，最后做同一RMS缩放。记录原始排列hash和最终buffer hash。重投影后不承诺保留原行范数或谱；对照检验同交互子空间内语义地址对应关系，而非完美保留全部几何。
- `off`：原 A 路径，不注册新参数/buffer，不扩展历史checkpoint契约。

零残差组、只有一个pair的码无需强制置乱；报告实际移动比例。若置乱重投影后退化或与原R逐位相同，拒绝将其视为有效负对照。

### 4. 一致的输入接口与因果性

在 TIGER 输入 embedding 边界增加可选变换接口，领域计算留在 tiger_catalog_grounded。不得用全局monkey patch或用forward hook猜测位置。

1. Encoder：历史输入按完整item的四层SID对齐，仅增强第二层；在插入SEP之前处理。padding和无效item不查表、不注入，不跨item取父码。
2. Teacher forcing：增强的是已输入的第二个SID token。BOS、第一/三/去重层不变；第二码目标的logit不能依赖第二码输入。当前decoder先拼BOS、最后截断输出，这个shift必须保留。
3. CGBS `_beam`：直接查SID表，不调用teacher-forcing forward，因此必须显式使用同一变换。仅prefix长度≥2时处理第二码，BOS不变。保留原inactive beam的合法占位行为，不为新模块改变top-k。

测试冻结encoder输出，改变未来目标suffix，检查较早decoder logit不变。eval模式下teacher forcing与同prefix的beam局部logit一致。decoder-only第二码注入不能直接解释第一/二码打分改善；encoder路径可能影响所有层。

### 5. 存储与身份

仅保存实际占用pair：sorted integer key=256*p+k，R为FP32。Beauty预计7091×128约3.46MiB；另加索引和metadata，不能将该数字当峰值显存。向量化lookup，禁止每样本CPU字典/同步。真实非padding未知pair报错；当前实验不解决目录动态更新。

领域模块由root持有单一状态；encoder/decoder通过显式调用接口使用，不在多个nn.Module路径重复注册。可选参数需进入optimizer且只出现一次，DDP梯度同步一致。

新增checkpoint契约含version、mode、injection_location=encoder_decoder_level2、fit权重/阈值/迭代数、catalog/PCA/SID/features/buffer hashes、RMS与局部随机seed。使用公共Artifact解析和lineage。推理加载已训练buffer/scale，禁止按已训练embedding重新计算q_A覆盖原值；加载前后验证身份，不兼容立即失败。

## Risks / Trade-offs

- 静态重建好不等于推荐好 → 主指标仍为验证集NDCG/Recall；不将目录MSE晋升为收益证据。
- 单例pair比例较大 → 分层报告，固定buffer不宣称消除记忆问题；不预设长尾提升。
- Transformer可能已学会交互 → additive与原A均是必要对照。
- 持续残差可能干扰行为适配 → 零起点和有界标量；无净收益即停止当前版本，不自动堆W/门控。
- CPU浮点hash可能受平台差异影响 → 记录构造环境，checkpoint冻结实际buffer；数值一致与逐字节一致分开报告。
- PrefixMem直接近邻 → 若有效再比较A+PrefixMem-style，清楚披露适配；不以少参数直接推导新颖性或速度。

## Migration Plan

以独立可选model组件和根脚本启用；原默认A不迁移，不覆盖历史checkpoint。新模型单独从头训练，不默认continuation。关闭功能回到原路径；不得将新checkpoint当旧A加载。

## Open Questions

推荐增益、encoder/decoder各自贡献、门值是否饱和、与PrefixMem的成本效果差异均待实验。单标量可能过于受限，但本版失败后须重新分析，不能在同一协议名下自动扩充参数。
