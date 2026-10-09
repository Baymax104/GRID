# 设计

继承既有无排除推理行为和消融身份校验，训练参数/冻结分组先按原 checkpoint 契约核验，再冻结推理模型。新增配置区分 method_version=v1 与 implementation_version=v5.3；Artifact alias 不作方法编号。

M1 增加 exclude_history（默认 true）及评价协议元数据，关闭时仍检查合法唯一 SID 和完整 keyed bundle，允许历史重叠，不进行 encode。各新 experiment/脚本通过统一 src.main；沿用 paper_ablation_copmrec_beauty、paper_mechanism_copmrec_beauty group。

运行前 CPU 测试恢复五臂并匹配原 raw dense 排序，验证错误来源拒绝；脚本 quoting/override/dry-run 与 Hydra compose 验证；Mutagen flush 后记录本地/远端字节、运行时 archive。每次推理独立 tmux、单张物理卡映射 cuda:0；复算不调用模型 forward。
