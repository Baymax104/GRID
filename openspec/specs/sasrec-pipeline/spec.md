# sasrec-pipeline Specification

## Purpose
将 SASRec 训练和推理接入统一 src.main/Hydra 入口、组件配置、根脚本及共享产物协议，规定安全参数透传、validation 选优和有限链路验证，交付可恢复且用户 key 对齐的预测。
## Requirements
### Requirement: Componentized official baseline configuration
SASRec SHALL 使用统一 src.main/Hydra 装配，路径 SHALL 由配置声明，模型/训练参数 SHALL 放组件配置。默认训练 SHALL 使用官方算法/Adam起点，明确 GRID step预算适配；checkpoint SHALL 只依据val/ndcg@10，训练 SHALL 不自动test。

#### Scenario: Compose train and inference
- **WHEN** 提供必须的data/catalog/dataset/device/checkpoint输入
- **THEN** resolved configs SHALL 正确装配 SASRec 数据与模型且不引用SID训练/内容embedding组件

### Requirement: Root script argument contract
脚本 SHALL 支持 --dry-run、两种notes语法、引用安全的路径、额外Hydra覆盖和多卡统一torchrun入口。错误flag、空必填项、非法seed/device/port SHALL 明确失败，训练 SHALL 默认不dry-run。

#### Scenario: Quoted values and override precedence
- **WHEN** 输入带空格/引号的路径或notes以及额外override
- **THEN** 参数 SHALL 保持单参数语义且用户override优先

### Requirement: Bounded real runtime verification
交付前 SHALL 使用统一入口完成本地dry-run和5 step逻辑验证，并核验checkpoint恢复与预测bundle，不把这些作为正式基线效果。

#### Scenario: Five optimizer steps and restored prediction
- **WHEN** 显式禁用外部发布并运行5 step及恢复推理
- **THEN** checkpoint SHALL 记录global_step=5、有限参数/optimizer状态，输出 SHALL 为用户key对齐的原始商品TopK bundle
