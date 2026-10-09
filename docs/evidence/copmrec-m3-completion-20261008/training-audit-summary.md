# CoPMRec M3 五臂训练完成审计（2026-10-08）

五臂均完成50,000 updates；5/5训练审计通过。审计为只读CPU模型恢复与W&B/文件核对，没有启动run或做forward。

| Issue | Variant | Run | Own-best step | Own-best raw val NDCG@10 | Last保存step | 50k状态完整checkpoint留存 |
| -- | -- | -- | -- | -- | -- | -- |
| BMX-122 | no_mixture | m3frgiim | 48500 | 0.0519791134 | 48500 | False |
| BMX-123 | no_residual | m364gahb | 47000 | 0.0473233201 | 47000 | False |
| BMX-129 | no_native | m3td2jxc | 41000 | 0.0533103906 | 41000 | False |
| BMX-142 | legal_generation | m3odfrrh | 46000 | 0.0515488088 | 46000 | False |
| BMX-143 | joint_ce_replace | m3kq1tuc | 43000 | 0.0524583235 | 43000 | False |

全部run finished、退出0、实际max_steps=50000终止、终值trainer/global_step=49999；100个validation点对应500…50000，Own-best为各自首次最高raw val/ndcg@10。存储last.ckpt不替代选点，checkpoint保存step与训练实际完成预算分别登记。

五份v0 checkpoint文件MD5与Artifact manifest匹配；SHA256、Artifact版本/digest、精确URI、组件冻结状态、完整AdamW和scheduler、cold残差0、catalog identity及CPU strict模型恢复全部核对。精确引用见training-audit-summary.json。

历史运行源码原始tar/manifest与304文件字节已核对，source42f0保持原值、origin verified。当前CPU审计代码source5e95单独登记，涉及诊断metadata序列化修复；不回填历史快照。首次审计过宽的源码范围检查已保留原始回执，并限定到实际训练/Testing恢复调用链。

Beauty525份数据分片和两份上游缓存SHA与启动前清单逐字一致，W&B input lineage/digest匹配。训练/验证原始history及指标范围完整保存在training-collected.json；仅登记原值与采样位置，不判断方法好坏或积极/消极结果。

Testing尚需独立启动与输出审计；本记录只完成训练及own-best准备，不代表消融issue完整交付。
