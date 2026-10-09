# CoPMRec v2 testing结果：当前相关性修正未转化为收益

日期：2026-10-04。训练run [beawrjef](https://wandb.ai/baymaxam/GRID/runs/beawrjef)，推理run [1gu2dsa1](https://wandb.ai/baymaxam/GRID/runs/1gu2dsa1)，Beauty/seed42，固定validation-selected best step42500。结论：本单元最终Recall/NDCG低于真实v0和LIGER dense；当前v2不满足推荐收益晋级条件。同checkpoint、同候选集合的诊断明确显示相关性重排对NDCG为负，不能将此次失利主要归因于新增漏召回，也不能据此断言整个d_ui路线无效。

## 结果身份与独立核验

- v2实际消费`copmrec_beauty_v2_train-checkpoint:v0`，producer beawrjef，文件MD5、内部step42500、v2 relevance契约与候选trace一致；lambda0.01/beta0.5、full testing、beam20、全部cold、Top10符合配置。
- 对照为真实v0推理 [6dspa7e3](https://wandb.ai/baymaxam/GRID/runs/6dspa7e3)，parent7y54j4m6/best48000；原LIGER推理 [042139al](https://wandb.ai/baymaxam/GRID/runs/042139al)，parent35ig0tz6/best45000。LIGER dense指标复算自同checkpoint trace保存的全目录content排名与dense Top10。
- 三组相同22363名testing用户、相同目标、相同catalog/input buffers与推理预处理。实际消费同一SID Artifact `rkmeans_inference_beauty-semantic-id:v0`（producer dq77e3wo）和embedding `sem_embeds_inference-semantic-embedding:v5`（producer3jtt9mpa）；目录12101商品、33 cold。
- 在node1只读核验Artifact manifest大小/MD5、producer、schema及预测bundle的keys/predictions契约，预测与trace Top10逐项相同。逐用户检查Top10无重复、SID合法；额外扫描真实testing TFRecord，独立构造user_id/目标SID，与全部trace标签一致。
- 从预测位置独立复算exact-SID Recall/NDCG@5/10，v2误差小于1e-12、candidate summary误差小于1e-9。v0有原始输出v2和producer-bound引用v4；二者bundle digest/size相同，使用producer-bound版本，保留历史修复边界，不以新引用补造原源码 provenance。
- 本次没有训练、推理或decoder search，只复算已有产物。历史baseline源字节审计沿用BMX-116已有证据，本轮不声称重新核验全部历史源码或训练seed总体。

## 最终testing指标

| 方法 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| 真实v0 hybrid | 0.043822 | 0.027383 | 0.071904 | 0.036448 |
| LIGER dense | 0.042526 | 0.027028 | 0.071413 | 0.036391 |
| v2融合hybrid | 0.038904 | 0.023538 | 0.069087 | 0.033285 |

v2相对v0：Recall10相对下降3.92%，NDCG10相对下降8.68%；相对LIGER dense分别下降3.26%、8.53%。所有数字来自相同Beauty/seed42用户；不混用三seed均值或validation指标。

固定checkpoint下，以用户配对bootstrap2000次、NumPy PCG64 seed42求未校正逐项95%区间：

| v2减对照 | ΔRecall10 [95%区间] | ΔNDCG10 [95%区间] | 新增/损失/净命中 |
|---|---|---|---|
| 真实v0 | −0.002817 [−0.005098, −0.000447] | −0.003163 [−0.004446, −0.001860] | 314 / 377 / −63 |
| LIGER dense | −0.002325 [−0.004561, 0.000000] | −0.003106 [−0.004459, −0.001758] | 351 / 403 / −52 |

v0比较的两项区间全负；LIGER比较的NDCG区间全负，Recall上界为0，不将后者称为明确严格负区间。区间仅描述当前已拟合checkpoint上的用户采样不确定性，不代表多seed训练不确定性。

## 推荐收益损失在哪里

以下均使用同一个v2 checkpoint。content-only为已记录的同自然候选content目标排名，不是重新训练一个删除ranking loss的模型。

| v2评分范围 | Recall@10 | NDCG@10 |
|---|---:|---:|
| 全目录content | 0.070205 | 0.035818 |
| 同beam+cold候选content-only | 0.069266 | 0.035615 |
| 同候选content＋相关性 | 0.069087 | 0.033285 |

相关性重排项的ΔRecall10为−0.000179，95%区间[−0.002013,+0.001745]；新增239、损失243、净少4个命中。ΔNDCG10为−0.002330，95%区间[−0.003518,−0.001174]。双方都在Top10的用户中，相关性令256人目标位置提升、651人下降。因此该项的明确负收益主要体现在已命中目标的前排位置，不是净命中数的大幅减少。

候选覆盖为v0 2461/22363=11.0048%、v2 2458/22363=10.9914%，净少3人，新增覆盖296、丢失覆盖299；总覆盖相近不代表候选集合完全不变。同v2的候选content相对全目录dense：ΔNDCG10=−0.000203，区间[−0.000671,+0.000242]。跨checkpoint的v2 dense相对v0 dense：ΔNDCG10=−0.000709，区间[−0.001701,+0.000278]。这两项均跨零，不能据此确认排序loss已显著破坏内容表示，或将全部差距归因于搜索。

NDCG差值有精确账目分解：

```text
v2融合 − v0最终
= (v2 dense − v0 dense)
  + [(v2候选content − v2 dense) − (v0最终 − v0 dense)]
  + (v2融合 − v2候选content)
= −0.000708527 −0.000124793 −0.002329848
= −0.003163168
```

同候选重排项占上述差值约74%；这是分数差值分解，不是联合训练中head或ranking loss的因果贡献比例。

修正仍满足实现的0.5×sigma上界，没有发现尺度越界：自然sigma均值3.7404，最终Top10修正绝对值均值1.3030、中位数1.3783，最大上界使用比例1.0。最终Top10中46.34%的修正绝对值超过其允许幅度的95%；该比例只统计最终Top10，不代表全部候选或训练样本。数值幅度受约束并未保护前排排序，较多tanh值接近边界是后续解释可参考的现象，不足以单独确认退化根因。

## 阶段判断

核心假设仍是生成表示与内容评分的联合学习能否转化为最终推荐收益。基础BMX-116的mass候选证据支持保留生成引导；此前v1.1有界loss校准只支持等权优化失衡，不证明相关性head或完整从零训练收益；旧冻结保护/高位干预没有建立总体净收益。此次v2单元直接未通过相对真实v0及LIGER dense的晋级门槛，同候选诊断明确否定“当前学得的相关性修正在本单元改善排序”这一主张。

因此当前v2不晋级为有效收益转化组件，保留负向证据，不将其包装为优于v0的方法。单seed结果不证明d_ui概念或整条论文路线不可能；联合训练损失、checkpoint选点差异等原因仍未完全分离。没有发现会使本单元结果失效的输出、标签、目录或checkpoint实现疑点，当前取舍不需要追加穷尽检查。此次新增实验/训练预算0，不自动扫描lambda/beta、不追加训练或第三臂；进一步迭代须有用户明确选择和新的、会改变决策的正面依据。

可复核证据：[W&B配置快照](evidence/copmrec-v2-testing-1gu2dsa1-20261004/wandb-runs.json)、[只读审计源码记录](evidence/copmrec-v2-testing-1gu2dsa1-20261004/audit-source.txt)、[完整复算与配对结果](evidence/copmrec-v2-testing-1gu2dsa1-20261004/results.json)。
