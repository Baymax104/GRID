# Beauty五seed冻结测试评估

## 目标与历史使用边界

本轮只评估A完整内容初始化相对匹配深层残差的泛化增量。复用seed42–46的十个best-val checkpoint，不新增训练，不运行范数候选，不用testing改方法、beam或checkpoint。

W&B直接data_split=testing查询为空，但扩展查询发现历史Beauty基线26qh50do、ye9u9yj7记录过test指标，jdaiqrpw、1bqxxkdn曾在data/beauty/testing推理；Sports也有对应历史记录。不能称testing从未被打开。用户本轮明确确认：这些testing结果仅用于报告基线，没有用于选择A、残差初始化或它们的超参数。因此表述为“独立于本轮A/残差方法选择的冻结测试评估”，披露基线历史使用，不称首次盲测。

十个训练run已重新在线核对finished、run_test_after_training=false、best-val checkpoint文件和Artifact digest。未发现这些A/残差模型既有testing记录。node1已只读确认evaluation/testing为两个实际目录且testing存在TFRecord文件；未进行逐样本时序/重复/标签泄漏审计，不能据目录不同证明所有数据独立性。未记录的离线使用无法由W&B排除。

## 冻结清单

完整URI、Artifact digest及代码hash见testing-frozen-manifest.json。精确run+role+file避免best/last与latest歧义。脚本配置testing_expected_checkpoint_digest为预期身份审计字段，实际Artifact digest必须在汇总时核对，不是下载时强制校验器。

| seed | A run / best step | 残差run / best step |
| --- | --- | --- |
| 42 | 5g3wpbg7 / 19000 | nq8he993 / 19000 |
| 43 | 0fuiq3vx / 19000 | wkpb2b4e / 19000 |
| 44 | o04nn9mf / 19500 | 79dk60vy / 19500 |
| 45 | kckgwr6j / 20000 | nlj0x573 / 20000 |
| 46 | v7q7y0jx / 19000 | ype8ur9j / 19500 |

SID来自4vyi4o6w，digest20f08b323a286fbb3f16b5ea27562af1；内容来自3jtt9mpa，digestab56af975eac589c27eed6094482cbd7。残差仍通过原训练对应组件加载已校验量化器来源，随后严格恢复推荐checkpoint。不训练任何模型。

## 手动启动：两个终端并行

两组各五次单卡推理，总计十次；CUDA_VISIBLE_DEVICES选择物理卡，逻辑devices始终[0]。不使用两卡DDP合并结果。

终端一：GPU0，全部A。

```bash
cd /data3/weizhenyu/projects/GRID
bash ./tiger_sid_initialization_testing.sh \
  --data-dir data/beauty \
  --condition full_content \
  --gpu 0 --master-port 29770 \
  --notes "Frozen A; seeds42-46; testing independent of method selection; baseline test use disclosed; no tuning"
```

终端二：GPU1，全部深层残差。

```bash
cd /data3/weizhenyu/projects/GRID
bash ./tiger_sid_initialization_testing.sh \
  --data-dir data/beauty \
  --condition deep_residual \
  --gpu 1 --master-port 29771 \
  --notes "Frozen matched residual; seeds42-46; testing independent of method selection; baseline test use disclosed; no tuning"
```

默认seed=all。每组失败立即停止，没有自动恢复/自动跳过完成项；补跑时加--seed 42至46之一、保持对应condition，只执行一项。不要重跑整组覆盖有效结果。只运行一个终端时可用--condition both（默认）在同卡顺序完成十项。

支持--dry-run、两种notes语法和额外Hydra override最后优先；正式命令不加dry-run。若override改变冻结协议，run必须标记偏离并排除本轮正式比较，不应根据结果决定是否纳入。

## 预先固定的汇总方案

先统一完成两组，不根据中途结果调整实验。以testing_protocol=beauty-sid-five-seed-v1筛选，seed与condition组成十个唯一单元。若重复完成run，核查来源和产生原因，不按指标择优。

1. 核对finished、run_mode=inference、testing、beam10、checkpoint实际lineage/digest、SID/content、残差量化器、初始化模式及目录fingerprint。两类标准writer发布recommendation_output和prefix_trace，不以inference summary缺少标量判断缺失。
2. 读取model output bundle与prefix trace。按用户key严格对齐，拒绝重复或缺失用户、标签不一致、非法/重复候选、预测非10项、最终prefix生存与命中不一致。核对十项全部用户集合及目标标签相同；必须记录每项用户数量。
3. 每个seed分别计算A与残差的NDCG@10、Recall@10及成对差；当前单目标Recall等同Hit。五seed等权平均，不挑seed、不把五份相同用户当成5N独立样本。报告成对差的样本标准差、胜负数和leave-one-seed-out平均差。
4. 用户层：每seed及固定五seed平均的用户差值，按目标前两层SID做prefix-cluster bootstrap，2000次、随机seed42、百分位95%区间。该区间条件于已训练的五组模型，只反映用户/前缀抽样不确定性。训练seed层另报告五个成对差的探索性t区间，不以用户区间代替训练重复；5个seed不足以给出强总体保证。
5. 前缀存活、命中增失及排名迁移作为解释性分析，不能在看到结果后挑子组作为主结论。NDCG@10主指标、Recall@10辅助指标，其余不改变最终去留依据。

## 判读与停止

- 若平均NDCG差为正、至少4/5 seed同向，leave-one-seed-out均值仍为正，且Recall没有多数seed同向退化，可视为通过跨数据集筛选门槛；仍需完整展示区间，若区间较宽则明确不确定，不据此宣称显著或机制已证实。
- 平均收益小且正负混合、对删除某个seed敏感或区间允许实际退化：降级为差别未明确，不自动追加seed/改模型。没有预设实用最小差异阈值，不事后发明阈值认定成功。
- 平均收益为负：不继续声称完整内容优于匹配残差。
- 测试通过后仅建议跨数据集确认，未经授权不启动；本轮不实施embedding预热，不把testing用于下一轮调参。若将来据testing修改方法，必须承认此划分不再是该新方法的独立确认集。

## 验证与交付

19项聚焦测试通过：Bash语法、十项映射、Hydra compose、testing路由、key/label/trace、单条件/单seed、空值/错误值、notes quoting、dry-run与override、单卡映射、失败停止。脚本十项与在线checkpoint文件/digest逐项一致。没有运行真实Trainer或完整推理。

OpenSpec严格验证通过。Mutagen flush成功，随后四个session均Watching for changes且无conflict；推理脚本已同步node1。未修改远端环境，未创建Git提交。
