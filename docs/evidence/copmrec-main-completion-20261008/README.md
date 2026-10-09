# BMX-116 九个正式单元完成核验

seed2026三个单卡Testing已在tmux正常结束，九个训练和九个Testing全部finished。九个训练50k与validation-selected best依据此前训练审计回执；本次逐一核对Testing消费的checkpoint Artifact digest、真实SHA256、版本/seed、SID/content及runtime源码。

| Dataset | CoPMRec NDCG@10 mean +/- std | LIGER hybrid NDCG@10 | Gain | CoPMRec Recall@10 mean +/- std | LIGER hybrid Recall@10 | Gain |
| -- | -- | -- | -- | -- | -- | -- |
| beauty | 0.04355895 +/- 0.00109371 | 0.02992889 | 45.54% | 0.07583956 +/- 0.00269898 | 0.05570213 | 36.15% |
| sports | 0.02398604 +/- 0.00054744 | 0.01574162 | 52.37% | 0.04356986 +/- 0.00067478 | 0.02864393 | 52.11% |
| toys | 0.04668824 +/- 0.00031526 | 0.02629864 | 77.53% | 0.07699705 +/- 0.00066238 | 0.04837214 | 59.18% |

三seed均值与样本标准差（ddof=1）。audit.json保存每seed四指标、用户配对bootstrap差值与95%CI：NumPy PCG64 seed42、2000次、未校正pointwise，不替代训练seed不确定性。

九个输出bundle的manifest MD5、SHA256、形状[N,10,4]和完整用户keys已核对；从原始testing TFRecord独立提取用户、末商品标签及最近20件输入历史。每用户10个合法唯一SID、CoPMRec历史重叠0，用户集合与标签一致。Recall/NDCG@5/@10用独立NumPy计算与W&B相差小于1e-8，用户数一致。testing数据逐文件哈希与用户/标签身份哈希保存在audit.json。

原正式LIGER hybrid九个输出仍复用，未重训、重评分或改协议；用户集合、SID/content及原四指标独立复算一致。原run的指标位于candidate_trace/hybrid命名空间，未用dense指标替换。原基线original hybrid/gen20/content ranking，保留其原历史政策；CoPMRec采用最近20件历史排除，二者历史资格不相同。audit.json记录每个基线历史重叠数，主表是完整正式方法比较，不能解释为仅网络结构或仅前缀机制的受控因果效应。

训练时未保留raw-data逐文件快照、last.ckpt较早状态等此前审计限制继续保留，训练有效不等于收敛证明。新Testing没有修复既往开发对测试集的可见性，不宣称blind holdout。各正式单元据协议有效而完成，Done不等于论文总体实验已完成；dense内部对照与消融仍另行待执行。

核验原始数据及baseline输出仅只读，无新训练、Validation、额外推理或模型更改。
