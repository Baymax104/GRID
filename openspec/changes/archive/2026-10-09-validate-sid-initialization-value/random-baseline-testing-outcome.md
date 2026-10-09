# 匹配随机初始化seed42 testing结果

## 结论

A相对mask_ce随机初始化的单seed收益保留到testing：NDCG@10相对+7.579%、Recall@10相对+8.562%，净多109个命中用户。按预先约定，本结果支持考虑补齐mask_ce seed43–46四次训练，复用已有A；本轮不自动实施或启动。不能据此恢复“完整内容优于残差”的主张，也不能把单seed用户区间当作跨seed证据。

## 身份和完整性

本轮W&B testing_protocol=beauty-random-init-seed42-v1恰好一项gfyz0026，finished。重新读取其与A sb6eapjd的config、notes、输入/输出Artifact记录，并下载预测与trace核验。随机checkpoint生产run y1dnupbt、文件checkpoint_epoch=000_step=019000.ckpt、digest14bee23f40882bef98b19685759afb35；A生产run5g3wpbg7，同19000步、digestaaba5b47dc9a5e94c0d23b75cf697238。

两项SID digest20f08b323a286fbb3f16b5ea27562af1及内容digestab56af975eac589c27eed6094482cbd7匹配。均testing、seed42、beam10、sequence_length180、四层SID/codebook256、单GPU32-true；data配置一致，无dry-run或predict截断。两种arm分别mask_ce和token_content_init，原匹配训练规则见random-baseline-audit.md。

22363个唯一用户的key和完整目标SID逐位一致；每用户10个合法唯一候选，目标合法；预测命中、排名和trace最终层一致，各层生存不发生失败后复活。目录fingerprint、split及checkpoint metadata一致。5196个目标前两层SID cluster。

## 指标

| 指标 | 随机mask_ce | A | 绝对差A-随机 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| NDCG@10 | 0.02965831661 | 0.03190617697 | +0.00224786036 | +7.579% |
| Recall@10 | 0.05692438403 | 0.06179850646 | +0.00487412243 | +8.562% |
| 命中用户数 | 1273 | 1382 | +109 | — |

当前每用户单目标，Recall等同Hit。A新增命中442人、丢失命中333人、共同命中940人。共同命中中A排名更好358人、更差307人、相同275人。不能将净增109解读为仅109用户受到影响。

按目标前两层SID cluster bootstrap，2000次、seed42，名义95%区间：NDCG差[0.00029995665,0.00421064839]，Recall差[0.00178260716,0.00818389334]。两者均正，但区间条件于这一对已经训练完成的模型，不包括训练随机性、checkpoint选择及事后研究选择的不确定性，不作稳健多seed显著性结论。

各层目标前缀生存人数：随机7856/2462/1547/1273，A8043/2591/1677/1382。聚合计数均更高，支持收益并非只有最终命中数量变化；幸存用户集合不同，不能据此认定每一层对每个用户都有改善或建立特定因果机制。

## 与既有证据的关系

同seed evaluation上A相对随机NDCG+6.426%、Recall+6.675%；此次testing同向+7.579%/+8.562%。这提高了“该初始化在Beauty seed42有实际收益”的可信度。但A相对匹配残差的五seed testing仍只有平均+0.752%、3胜2负，二者不可混为同一主张。

本比较在看到A/残差testing之后才提出，属于探索性补充；历史testing用于基线报告的用户说明仍保留，不称首次盲测或预注册确认。

## 最小后续建议

只补mask_ce seed43–46四个匹配随机初始化训练，原A不重训，保持既有20k预算、每500step validation、best-val选择和所有训练参数。随后只补对应四项testing推理，与同seed既有A配对。预先锁定全部四项，不按途中结果挑seed，不调A、不加入预热或范数改动。最终报告五seed平均差、胜负、差值分布和用户条件区间；该复核属于已有方法的补充稳健性实验，不能抹去testing已观察的历史。

在完成前，不用随机seed42与A五seed均值比较；不称内容初始化稳定有效；也不认为补足有效性对照自动解决论文新颖性。

原始W&B快照、全部Artifact digest和逐用户CSV保存在tmp/random_testing_results；analysis.json含本轮统计。只读脚本为tmp/inspect_random_testing.py与tmp/analyze_random_testing.py。本轮未修改生产代码、未启动新训练或推理。
