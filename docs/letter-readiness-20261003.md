# LETTER 可运行准备核验（BMX-58）

## 范围

完成独立 LETTER 可运行准备及 Beauty/Sports/Toys × seed42/200/2026 九槽位命令，正式实验由用户手动启动。此记录不证明论文效果；有限验证产物不能作为正式实验上游。

## 实现与来源

LETTER 作者仓库固定 `8d0154e28de37dbb6e24871c508ad8ddb1921cda`。新增独立32维CF teacher，依据 [SASRec 官方公式](https://github.com/kang205/SASRec/tree/e3738967fddab206d6eeb4fda433e7a7034dd8b1)；不调用项目 SASRec、RQ-VAE、TIGER、LIGER、CoPMRec 模型。作者没有发布完整CF teacher训练代码，因此其history50、2blocks/1head/dropout0.5、单卡batch128、Adam0.001、50k step/1000 step选优预算明确登记为GRID适配。没有新增超参数搜索。

新增统一入口 `letter_cf_train` / `letter_cf_export`，复用公共 reader、FileDataModule、MetricEngine、artifact resolver及writer；CF训练只从training产生梯度与负样本排除集合，evaluation用于选checkpoint，不消费testing。CF导出原始商品key及未归一化32维商品表；原始商品0映射非零模型ID，模型0只作padding。冷商品保留，缺少正交互监督但可能收到负采样梯度。

## 真实固定内容与数据

| Dataset | Producer / Artifact | 内容shape | 商品 | evaluation/testing用户 |
| --- | --- | --- | --- | --- |
| Beauty | 3jtt9mpa / sem_embeds_inference-semantic-embedding:v5 | 12101×1024 | 12101 | 22363 |
| Sports | psec3u5i / sem_embeds_inference-semantic-embedding:v6 | 18357×1024 | 18357 | 35598 |
| Toys | d1q00dco / sem_embeds_inference-semantic-embedding:v7 | 11924×1024 | 11924 | 19412 |

三个实际bundle重新解析并核验有限值、文件SHA256、排序后的key hash，与共同RKMeans目录keys逐项一致；只比较keys，LETTER不消费RKMeans SID数值。正式Tokenizer命令显式 `input_dim=1024`，768仅用于初次synthetic验证和组件通用默认。

重新读取三个数据集training/evaluation/testing全部分片：分别175/279/152分片；所有商品在共同目录，所有行至少两个商品，每split用户无重复，三个split用户集合逐项一致。登记path/size/file SHA256构成的manifest和user key hash；当前hash采用明确新序列化规则，不冒充旧审计不同算法的hash。

内容文件SHA256：

- Beauty：`968b7fa491bd8ef6b2fe58942b45da10dbd788882ddf4e9a9de3b6cf1d0e11b1`
- Sports：`299daea3ee7bfc52ab552d82615394679a4824ee67a4d02a23a8a315d9968513`
- Toys：`28547fac39885e25afe3518492afa99938ca50dec2c004a3387532cee5db4432`

## 已完成验证

- 聚焦模型/数据/配置/共享Artifact及命令测试合计62 passed，包含Bash语法、九槽位63个实际命令块stub、quotes、带等号checkpoint、notes、master port、GPU进程数与override顺序，stub不会启动训练。
- CF teacher block与独立NumPy公式对照、因果attention、完整training行负采样排除、key0、embedding梯度、导出CPU及错目录checkpoint拒绝通过。
- node1既有Python3.11.10 / PyTorch2.9.1+cu128 / CUDA12.8 / Lightning2.6.5，未修改虚拟环境或Torch/CUDA。Mutagen flush成功，三个session均Watching for changes、无conflict。
- 九个CF训练dry-run：真实固定内容目录，默认4workers/batch128/FP32，物理GPU7，max_steps1全部exit0且loss有限；无W&B业务发布，无validation/testing消费。
- 三个CF有限训练：max_steps5，evaluation限制两批64用户，不消费testing；按有限validation best导出全目录，输出12101/18357/11924×32，与内容keys一致且有限。此次实际保存的best为step2；已核验其global_step与optimizer state均2、exp_avg非零、参数有限、导出与该checkpoint商品表逐项完全相等。不能把step2保存证据描述为step5 checkpoint核验。
- 三个目录4×256初始化探针：零学习率、max_steps1，代码采用作者n_init10/max_iter10/n_jobs10。encoder/decoder逐参数等于seed42原始初始化，仅初始化码本，不宣称码本训练收敛。三个全目录SID合法、唯一，shape分别12101/18357/11924×4；同CUDA/medium精度重复导出完全相同，前三码保留，末码分别改变36/142/56个。初始nearest唯一数量12080/18260/11883，最大prefix人口4/14/12。
- 九个推荐DDP训练dry-run：物理GPU6/7、每卡batch128/global256、默认4workers/FP32、max_steps1，全部exit0且loss有限；使用初始化探针SID，禁用validation/testing和业务发布。不声称九个seed的正式Tokenizer已训练，探针仅按每数据集seed42初始化。
- Beauty真实目录推荐五步两卡训练：global_step5、optimizer所有state step5、exp_avg非零、参数有限；evaluation每rank两批共128用户。按本次有限validation选出的step5 checkpoint恢复，仅对evaluation推理，每rank两批合并128×10 CPU bundle；用户属于evaluation、候选全部合法且每用户10个互异。按原始evaluation目标独立重算Recall/NDCG@5/@10均0，不能作为性能结果。未验证在线W&B summary，未消费testing。

## 碰撞适配与失败边界

Beauty有限五步训练完成但导出失败；保存checkpoint的CPU诊断nearest仅20个SID，最大三码prefix8496，不能用末层256容量强行唯一化。另两个五步模型的导出也因prefix超容量明确失败，失败日志保留。上述五步结果不证明正式训练收敛，也不登记为正式SID。

作者20轮末层Sinkhorn使用逐行argmax，不能保证硬码唯一，作者脚本容许残余冲突；GRID原始商品级完整目录需要唯一SID。新增 `letter-sinkhorn-prefix-assignment-v1`：作者式修复后，对仍碰撞的三位prefix**全部商品**（含已占末码的邻居）求末层最小总残差距离一对一分配；只改末码，不加第五位。prefix人口超过256仍失败，不通过扩大码空间掩盖未就绪码本。正式训练后的SID必须重新过门禁，不能使用探针产物替代。

此前初始化n_jobs1的串行验证成本较高，现按作者n_jobs10冻结。64×8固定seed重复并行结果逐项一致，但串/并行中心与分区不同，不能声称数值等价；完整运行源码分别保留。统一入口采用既有matmul_precision=medium；CPU/CUDA编码比较曾产生差异，SID重复性只在同CUDA/medium设置验证，不据此改公共精度协议。

## 命令和选优

[九槽位完整命令](letter-issue-commands-20261003.md) 为每槽位提供CF teacher训练、CF导出、Tokenizer训练、SID导出、推荐训练、Testing、显式dry-run；与其他实验issue一致，显式CUDA_VISIBLE_DEVICES/NPROC_PER_NODE/MASTER_PORT、逐参数换行、W&B group和issue/dataset/seed notes。

各阶段未来产物使用带shell守卫的 `LETTER_CF_BEST_CKPT`、`LETTER_CF_EMBEDDING`、`LETTER_CF_SOURCE`、`LETTER_TOKENIZER_BEST_CKPT`、`LETTER_SID`、`LETTER_BEST_CKPT`。从本单元实际producer记录固定版本/文件/metadata，不虚构尚未生成的URI。CF/推荐按val/ndcg@10 max，Tokenizer按val/collision_rate min，last只用于恢复。每seed分别生产全链路。

准备检查使用的CF和SID均显式非正式；正式命令从本单元CF训练开始，不使用这些验证产物。正式50k推荐及20000epoch tokenizer未启动，W&B训练选优/最终Testing lineage及指标仍待用户启动后审计。temperature固定1，不据此声称rank强化机制收益。

## 证据位置

node1根目录 `logs/letter-readiness-20261003/` 保存content-audit.json（含Artifact digest/文件MD5与manifest一致）、split-audit.json、cf-verification.json、cf-dry-runs.json、sid-verification.json、recommendation-dry-runs.json、recommendation-verification.json及源码核验JSON。九个 `<dataset>-seed<seed>/{cf-dry,recommendation-dry}/` 保存日志与Hydra配置；三个 `<dataset>-bounded/{cf-train,cf-export,tokenizer-initialization-probe-parallel,sid-initialization-probe-parallel}/` 保存通过的探针产物，Beauty额外有recommendation-five-updates和evaluation-predict。日志为UTC 2026-10-02，本地为2026-10-03；本地logs保存JSON副本。失败与停止的串行探针日志原样保留，不作为成功证据。

CF有限训练/导出的运行时源码指纹 `c530b279adf7d706c5f23e1bce706a354d035c177606c65b758c1c4fa446c399`；通过的初始化/SID/推荐训练与恢复指纹 `710096cf6eb19b9be00d2c1cabfa6f711ce4eabfe838a90e98cc9982fbf0c537`。两版14个metadata/source_snapshot归档分别保留dirty/untracked实际字节、manifest/runtime，local-workspace来源verified；每个304文件大小/hash逐项通过，不将后版来源追补为前版。dry-run按统一launcher不保存业务归档，仅记录Hydra配置/日志。

OpenSpec新增 `add-letter-cf-teacher`、`fix-letter-sid-collision-export`；原五个LETTER change保留，七个分别strict验证，未合并大提案。Linear BMX-58准备完成，九槽位转Todo/待用户正式运行；BMX-130仍Todo，正式结果尚无。
