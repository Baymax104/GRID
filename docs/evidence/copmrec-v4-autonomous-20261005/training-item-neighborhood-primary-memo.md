# 训练交互商品邻域的原论文定位（只读）

本备忘录仅定位两个成熟的协同推荐方法，不构造评分、forward、预测或新实验。固定 pool 已未通过整体双10%门槛；这不构成邻域机制的新问题证据。是否存在与当前漏命中/错曝光有关的可区分训练关系，仍等待 architecture 的实际只读统计。本文件不提出参数、实验额度或新颖性/收益主张。

## 两个原始方法

**Sarwar 等，WWW 2001《基于商品的协同过滤推荐算法》。** 输入为用户—商品评分矩阵；先找两商品的共同评分用户，计算商品相似度，再用目标用户实际评分过的邻居商品预测目标商品。论文比较 cosine、相关系数和 adjusted cosine；后者减去各用户平均评分，校正用户评分尺度差异。weighted sum 按商品相似度加权该用户的评分，并以相似度绝对值之和归一化。因此不同历史产生不同预测；中心化不等同热门惩罚，也不能未说明就照搬为二元隐式交互规则。[共同评分与相似度](https://www.ra.ethz.ch/CDstore/www10/papers/519/node11.html)、[adjusted cosine](https://www.ra.ethz.ch/CDstore/www10/papers/519/node14.html)、[weighted sum](https://www.ra.ethz.ch/CDstore/www10/papers/519/node16.html)。

论文将预计算商品相似度、存储少量邻居再与用户已购商品求交称为 model building；离线构建与在线预测职责分开。邻域存储量与质量有 trade-off，原任务的评分预测结论不提供当前 full-catalog 下一商品 R10/N10 保证。[原论文 §3.3](https://www.ra.ethz.ch/CDstore/www10/papers/519/node18.html)、[作者所在 GroupLens 的原刊 PDF](https://files.grouplens.org/papers/www10_sarwar.pdf)。

**Linden、Smith、York，IEEE Internet Computing 2003《Amazon.com 推荐：商品到商品协同过滤》。** 原文按商品→购买该商品的用户→这些用户购买的其他商品记录共购，离线形成相似商品表。商品向量的维度对应购买用户，cosine 是文中给出的常用相似度；在线按当前用户购买/评分商品查邻居并聚合。它依用户历史或购物车条件生成结果，并非对所有用户加同一个商品截距。[原刊 PDF，pp.78–79（高校镜像，内容为原论文）](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)。

若将商品用户向量解释为二元购买指示，cosine 可数学化简为共购用户数除以两商品购买用户数的几何均值。这是解释式推导，说明尺度归一化区别于裸共购次数；不能据此称热门偏置已消失。特别注意：pp.76–77 的显式 inverse-frequency 讨论属于传统 user-CF，不能偷换成该文 item-item 方案的已指定规则。原文也说明离线表构建有时间和内存成本；未提供本仓库的完整融合公式或效果承诺。[同一原刊 PDF，pp.76–79](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)。

## 当前任务中的解释边界

两篇原文支持的成熟定位是“用训练交互建立商品关系，再由该用户的历史选择并聚合关系”。商品关系表可以共享，个人分数仍随历史变化；全局商品 bias 则在所有历史上施加相同截距。CoPMRec 已有历史编码，不能据这种定位声称它没有个性化、已经缺失该关系，或邻域一定会补救现有 bad cases。

原用户共购/共评、逐条原始交互次数与 causal prefix→target 次数是不同统计对象。若解释当前 GRID 数据，必须先说明去重、重复商品和因果截止的真实定义；不得混入 Evaluation/Testing 目标，也不得将当前重复展开的训练样本次数伪称原论文的独立共购用户数。当前仅等实际统计区分相关性，不生成新的目录分数。

## 零梯度与预算

确定性 count aggregation 没有 T5/CoPMRec optimizer 更新，因此不是一次新的梯度训练。但从训练交互估计并保存共现/相似度/邻域表属于数据驱动的协同结构拟合与离线模型/索引构建；Sarwar 原文也明确称为 model building。若日后将其用于推荐评分，不能称为只运行既有冻结模型、不能以“无梯度”隐藏输入、统计规则、构建耗时、存储和新组件选择成本。[原论文 §3.3](https://www.ra.ethz.ch/CDstore/www10/papers/519/node18.html)。

本备忘录未进行拟合或推理，不决定任何新增构建/实验如何计入正式额度。累计 **5 次训练 / 30000 steps 已用尽**的事实不变；没有自动追加预算、恢复已关闭阶段或开启新评分/验证的授权。文献定位与任何后续路线决策分开，fresh 正面问题证据尚待实际统计。
