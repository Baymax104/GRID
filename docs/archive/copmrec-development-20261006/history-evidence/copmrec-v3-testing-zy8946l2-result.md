# CoPMRec v3 Testing结果：首层遗漏改善，整体尚无净收益

后续用户决定：保留当前首层Max恢复方案，继续后层bad case分析并探索v3.1，见 [探索报告](copmrec-v3-1-exploration.md)。下方整体@10未晋级结论保持，不作为撤销首层组件的决定。

日期：2026-10-04。用户训练run [rbha00vx](https://wandb.ai/baymaxam/GRID/runs/rbha00vx)，推理run [zy8946l2](https://wandb.ai/baymaxam/GRID/runs/zy8946l2)，Beauty/seed42、validation-selected best43500、单卡Testing。独立复算通过：v3的@5点估计提高，主评价@10低于真实v0及LIGER dense，配对95%区间均跨零；当前单元没有建立推荐净收益，也不能确认稳定退化。首层剪枝的局部预测得到观测支持，后续全局beam竞争仍丢失内容可识别目标。

## 结果身份与核验

- 实际消费 `copmrec_beauty_v3_train-checkpoint:v0`，producer rbha00vx，digest `48204b0d67bf2ba6c244d6f1b09d583a`，本地文件MD5/大小与manifest一致，内部step43500、v3版本及[max,mass,mass,mass]目标契约正确。alpha=0.668786346912384来自保存参数，与candidate/path metadata一致；v0 alpha约0.8132556080818176。
- 实际单卡逻辑[0]、完整testing、beam20加全部cold、Top10 content终排；两个trace writer均启用，没有标签注入候选。原输入SID/embedding的producer/digest与训练及两组对照一致。
- 对照为真实v0推理 [6dspa7e3](https://wandb.ai/baymaxam/GRID/runs/6dspa7e3)，parent7y54j4m6/best48000；原LIGER推理 [042139al](https://wandb.ai/baymaxam/GRID/runs/042139al)，parent35ig0tz6/best45000。LIGER dense来自其全目录content trace，而不是该run的hybrid summary。
- 三组相同22363名testing用户/真实目标，相同12101商品目录及33 cold、相同catalog输入buffers和预处理。重新读取真实testing TFRecord，独立构造user_id与最后目标SID；全部与trace标签相同。预测bundle为keys/predictions，shape[22363,10,4]，所有SID合法、每用户无重复，预测与candidate trace一致。
- Artifact producer、文件大小/MD5、trace schema及全部candidate summary均通过；独立exact-SID Recall/NDCG@5/10与W&B误差小于1e-12。v0等价输出版本使用producer-bound引用，保留历史引用修复边界，不以新引用补造早期源码provenance。
- v0/v3真实path frontier与生成SID、目标存活/首个失败层一致。v3实际code Artifact三个文件摘要、manifest及archive SHA256正确，归档25个相关运行文件与交付源码匹配，source_sha256=`ab9d0d4073a7545ab763e2da739c180c473c8853d16a6bfda0d02a238f2eacd1`。
- 本次只读取已有产物，无训练、前向、decoder search或新推理；baseline历史源字节边界沿用已保留审计，不宣称本次重验全部训练seed或历史源码。

## 最终Testing指标

| 方法 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| 真实v0 hybrid | 0.043822 | 0.027383 | 0.071904 | 0.036448 |
| LIGER dense | 0.042526 | 0.027028 | 0.071413 | 0.036391 |
| v3 hybrid | 0.045566 | 0.027886 | 0.069892 | 0.035701 |

v3相对真实v0的Recall10/NDCG10分别为-2.80%/-2.05%，相对LIGER dense分别为-2.13%/-1.89%。@5点估计有所提高；不能忽略@10损失或用局部改善宣称总体收益。

固定checkpoint下，按用户配对bootstrap2000次、NumPy PCG64 seed42，求未校正逐项95%区间：

| v3减对照 | ΔRecall10 [95%区间] | ΔNDCG10 [95%区间] | 新增/损失/净命中 |
|---|---|---|---|
| 真实v0 | -0.002012 [-0.004248, +0.000045] | -0.000747 [-0.001867, +0.000252] | 264 / 309 / -45 |
| LIGER dense | -0.001520 [-0.003711, +0.000447] | -0.000689 [-0.001729, +0.000303] | 253 / 287 / -34 |

两组两项区间均跨零，仅反映当前拟合checkpoint上的用户采样不确定性，不代表多seed训练不确定性。v3相对v0共同命中中518人升位、406人降位；位置贡献+0.000153647，新增命中贡献+0.004351140，丢失命中贡献−0.005251643，相加为NDCG差值−0.000746855。主要损失账目是命中交换，没有出现v2式的共同命中位置总体恶化。

## 候选收益与损失

| 项目 | v0 | v3 |
|---|---:|---:|
| 目标候选覆盖人数 | 2461 | 2399 |
| 目标候选覆盖率 | 11.0048% | 10.7275% |
| 各自dense Top10目标未入候选 | 97 | 114 |
| 其中首层失败 | 67 | 0 |
| 第2层失败 | 25 | 73 |
| 第3层失败 | 3 | 29 |
| 第4层失败 | 2 | 12 |
| 同checkpoint hybrid相对dense新增命中 | 90 | 103 |
| 同checkpoint hybrid相对dense损失命中 | 97 | 114 |

跨checkpoint新增覆盖387、丢失覆盖449、净少62；最终Top10新增264、损失309、净少45。平均每用户生成beam20中共享12.904个商品，候选集合有明显交换，而不是单纯增加候选数。

v0原97个dense Top10遗漏中，v3恢复55个到候选、35个到Top10；其中原首层失败67个恢复47个候选、29个Top10。其余18个虽覆盖，但v3内容位置已在10名以后；仍首层失败的20个也不再属于v3 dense Top10。因此“v0的旧案例恢复”与“v3自身内容可识别目标可达”须分别报告，不能把两个不同checkpoint的目标集合视为固定。

两个checkpoint都将目标排入dense Top10的1325名用户中，v3恢复32个v0遗漏、丢失53个原v0命中，候选导致净少21个命中；这说明首层改善同时存在后续搜索损失。该分组为事后描述，不是固定模型的因果干预。

v3自身114个dense Top10遗漏全部发生于2–4层；失败时局部分支rank均≤20，其中70例rank≤3，51例失败处合法子节点总数≤20。父前缀仍在上一层frontier，目标分支在跨父节点的累计分数beam20竞争中被淘汰；剩余问题包含全局竞争，不能仅解释为每个父节点的局部候选不足。局部rank按strictly_greater_plus_one计数，保留tie口径。

## 具体案例

| user / 目标商品 | v0观测 | v3观测 | 解释边界 |
|---|---|---|---|
| 7987 / Aztec Secret泥膜，item789 | dense第1，首层局部rank23淘汰 | dense第1，首层rank2，最终第1 | 内容位置保持，首层可达性恢复 |
| 905 / Aveeno沐浴露，item421 | dense第1，首层rank45淘汰 | dense第11，首层rank13，最终第9 | 恢复了旧案例；内容位置也变化，不能视为同评分干预 |
| 21319 / TS-2直发夹，item1178 | dense第1、最终第1，逐层rank[2,1,1,1] | dense仍第1，逐层rank[2,2,0,0]，第2层淘汰 | 首层未丢，但后续全局beam竞争失去最强content目标 |
| 8667 / Jergens身体乳，item942 | dense第2、最终第2 | dense第1，首层rank2、第2层rank4后淘汰 | 内容能力仍在，候选剪枝造成漏召回 |
| 20779 / Gelish指甲油套装，item674 | dense第7，第3层失败 | dense仍第7，第2层失败，局部rank9 | 旧后层失败未修复，而且淘汰提前 |

21319目标的前两层mixed累计log概率从v0约−4.09415变为v3约−4.88928；8667从约−4.30750变为−5.04257。trace没有保存所有被扩展分支的完整打分阈值，本次只确认路径及条件分布变化，不据此指定唯一原因。商品标题来自已保留原数据分析记录，仅用于辨认案例，不参与推理。

## 差距分解与阶段判断

| 同checkpoint评分范围 | Recall@10 | NDCG@10 |
|---|---:|---:|
| v0全目录dense | 0.072218 | 0.036526 |
| v0候选content | 0.071904 | 0.036448 |
| v3全目录dense | 0.070384 | 0.035782 |
| v3候选content | 0.069892 | 0.035701 |

NDCG的精确账目：

```text
v3最终 − v0最终
= (v3 dense − v0 dense)
  + [(v3最终 − v3 dense) − (v0最终 − v0 dense)]
= −0.000744453 −0.000002402
= −0.000746855
```

绝大部分点估计差值对应dense排名部分，两个模型各自“候选相对dense”的NDCG净贡献几乎相同。dense差值95%区间[−0.001766,+0.000289]、候选转换差值区间[−0.000649,+0.000625]均跨零；这不是聚合训练损坏content的因果证明，也不能忽略114个确实可由trace验证的搜索遗漏。聚合、联合训练得到的内容/生成参数及alpha均变化，单一跨checkpoint比较无法完全隔离原因。

当前核心预测是“首层Max改善内容可识别目标的可达性，并能在同预算下转为最终净收益”。首层局部现象与预测一致，整体净收益门槛本单元未通过；在各自best、同数据与beam20下没有建立优于真实v0/LIGER dense的效果。保留首层恢复和后层失败证据，当前v3暂不晋级；不由这一个seed否定整个候选路线，也不把@5或个别案例改善作为扩预算依据。

既有v0矩阵的局部候选支持、v1/v1.1未形成完整收益和已关闭v2负向结果边界保持。当前记录的是第二层跨父前缀竞争这一未决候选问题，没有引入新模块、修改模型、分配额外预算或启动实验。testing已用于开发分析，后续不能包装为完全独立确认。

证据：[W&B快照](evidence/copmrec-v3-testing-zy8946l2-20261004/wandb-runs.json)、[只读审计源码](evidence/copmrec-v3-testing-zy8946l2-20261004/audit-source.txt)、[完整复算和全部案例](evidence/copmrec-v3-testing-zy8946l2-20261004/results.json)、[旧案例转移](evidence/copmrec-v3-testing-zy8946l2-20261004/case-transitions.json)、[路径案例与双dense可识别分组](evidence/copmrec-v3-testing-zy8946l2-20261004/focused-path-analysis.json)。
