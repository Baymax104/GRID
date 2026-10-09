## Context

作者源码固定 8d0154e28de37dbb6e24871c508ad8ddb1921cda。用户要求模型独立，只复用框架公共组件。

## Goals / Non-Goals

实现独立 tokenizer 及可比较的公式/梯度测试；不包含推荐模型、CF teacher 训练、数据或运行配置。

## Decisions

- nn.Module 核心，后续由独立 Lightning wrapper 装配。每层 VQ 为 codebook MSE + 0.25 commitment MSE + beta diversity，各层平均；CF 为 batch dot-product CE。
- 保留作者逐层 STE/残差相减的实际梯度；不擅自修正作者实现。参数 alpha/beta 与论文表述独立记录。
- constrained K-means 初始化及每 epoch codebook 分组，使用正式依赖，不替换为普通 K-means。n_init10/max_iter10，按作者n_jobs10并行restart；固定seed小样本重复并行结果相同，但不同于早期串行结果，不宣称串/并行数值等价。随机 positive 排除 self，单成员组显式失败。
- 编码不计算随机 loss；导出对全目录最近邻 SID 做末层 Sinkhorn 碰撞修复，最多20轮。后续 `fix-letter-sid-collision-export` 增加明确的GRID容量内末层硬匹配，超容量仍报错，不添加第五位。
- checkpoints 包含 initialized flag 与 cluster labels；RNG 使用框架种子。

## Risks / Trade-offs

- 官方固定256/10限制推广到小测试尺寸，但正式尺寸和公式不变。
- 作者未锁定 sklearn 及 CUDA RNG；声明数值/梯度容差一致，不宣称跨环境逐位相同。
