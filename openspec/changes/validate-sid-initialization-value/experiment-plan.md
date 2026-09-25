# SID 初始化的效果检验

## 本轮回答的问题

完整内容条件均值与残差质心存在深层几何差异，但收益未明。本轮固定A作为参照，不引入新模块，只新增以下两次训练：

| 条件 | 第1层 | 第2、3层 | 作用 |
| --- | --- | --- | --- |
| 既有A：5g3wpbg7 | 完整内容均值 | 完整内容均值 | 复用参考，不默认重跑 |
| first_only | 完整内容均值 | 随机 | 检验深层初始化必要性 |
| deep_residual | 完整内容均值 | 均值/整体尺度匹配后的残差质心 | 检验深层完整内容几何的实际价值 |

去重层随机，初始化后所有token正常训练。两条件使用与A相同的合法前缀CE、模型、每卡batch128、lr0.0005、Beauty seed42、20k steps、每500step验证及best-val NDCG@10选择规则。无内容评分分支或辅助损失。

映射公式和归因边界见 [design.md](design.md)。此残差对照匹配A的逐坐标均值及总中心化能量，不能称为原论文完整复现，也不能声称排除了逐物品预处理等所有因素。

## 启动命令

在 node1 仓库根目录 `/data3/weizhenyu/projects/GRID` 执行：

```bash
bash ./tiger_content_initialization_train.sh \
  --data-dir data/beauty \
  --condition both \
  --seed 42 \
  --notes "Test whether deep full-content SID initialization is useful; first-only versus matched residual centroids; reference A=5g3wpbg7"
```

脚本默认依次运行first_only和deep_residual，各20k steps，均使用物理GPU0、1和两个torchrun进程。第一项失败会停止，不启动第二项。不要在环境外额外启动另一份both。

### 两组GPU并行执行

如果GPU0、1和GPU2、3均空闲，可以在两个终端分别执行以下命令，不运行上面的both命令：

```bash
bash ./tiger_content_initialization_train.sh \
  --data-dir data/beauty --condition first_only \
  --gpus 0,1 --master-port 29750 --seed 42 \
  --notes "Deep SID initialization ablation; first layer only; reference A=5g3wpbg7"
```

```bash
bash ./tiger_content_initialization_train.sh \
  --data-dir data/beauty --condition deep_residual \
  --gpus 2,3 --master-port 29751 --seed 42 \
  --notes "Deep SID initialization comparison; matched residual centroids; reference A=5g3wpbg7"
```

两个实验没有结果依赖，使用不同通信端口和不同task_name输出目录。`--gpus`选择物理GPU，Trainer内部的devices仍为映射后的[0,1]；无需手动改成[2,3]。每项依然两个进程、每卡batch128、20k steps，与顺序执行时的实验条件一致。物理GPU记录为`initialization_gpu_ids`。并行运行可能共享CPU/I/O资源，实际总耗时不保证减半。

并行参数补充验证：26项相关测试通过，覆盖两种参数语法、两组GPU映射、独立端口、默认串行兼容、空值/重复GPU/非法端口及脚本语法；未启动真实GPU任务。

若第一项已完成，只运行第二项时将 `--condition both` 改为 `--condition deep_residual`；只运行首项则使用 `--condition first_only`。脚本不是自动resume管理器，重复both会重新启动两项。

`--dry-run` 可显式追加并透传统一入口；它属于统一入口的实际smoke执行，不等于只打印命令。本次agent仅做内存单测/配置compose/脚本截获检查，没有运行dry-run或完整实验。额外Hydra override优先级最高；若改变seed/预算/模型等，后续分析必须按resolved config判断可比性，不能只读condition标签。

## 固定来源

2026-09-17通过W&B只读API及现有选择器复核，三组run/role/file均唯一匹配：

- SID：`4vyi4o6w` → `rkmeans_inference-semantic-id:v2`，digest `20f08b323a286fbb3f16b5ea27562af1`。
- 内容：`3jtt9mpa` → `sem_embeds_inference-semantic-embedding:v5`，digest `ab56af975eac589c27eed6094482cbd7`。
- 量化器：`ofvhli6c` → `rkmeans_train-checkpoint:v3`，digest `312fbcb197a710aaa6f40f36d1e0a08f`。
- 量化器文件SHA256：`12ff2897eceab16780b983d5ebf4360d139c973f5adb0c3dd1bfa3f010307649`。标记遵循真实保存格式：顶层`layers_initialized`布尔列表，码本位于`state_dict`。

量化器引用属于初始化输入，不是推荐模型恢复路径；`ckpt_path=null`保证从头训练。两个条件使用不同task_name，W&B记录`initialization_protocol`、`initialization_condition`及参考run。新模式及码本tensor hash进入checkpoint契约；默认A契约不变。

## 完成后的判读

先核实run finished且训练至20k，核对配置、来源、每卡batch及checkpoint规则。比较最佳NDCG@10、对应Recall@10、完整验证曲线、共同step19000及最终共同步骤。

1. A不优于first_only：没有证据说明深层必需，优先考虑简化。差距小或曲线冲突则保留不确定，不宣称统计等价。
2. A优于first_only，但不优于deep_residual：深层语义初始化可能有价值，完整内容代表量的独特贡献仍未成立。
3. A同时优于两者：值得将深层完整内容几何作为候选机制，后续再限定推理配对比较和seed复核。单seed最佳验证点不足以确认论文结论。

不因负结果自动追加模块、调参网格或更换假设。下一步范围由这两个结果决定。

## 轻量验证证据

- 受影响模型、配置/脚本、artifact回归：232 passed；修正真实checkpoint标记格式并补两个配置实例化测试后，相关模型/loader测试31 passed。
- Ruff通过；OpenSpec严格校验通过；脚本语法、notes两种语法、quoting、空值、错误输入、override优先级、双卡环境及失败即停均由截获调用测试验证。
- 真实12101物品缓存的静态初始化通过：固定256项×3层SID身份全部一致；首层逐位一致；随机流不变；后两层均值最大误差约1.2e-7，标准差匹配。
- 原始证据：`tmp/initialization_comparison/producer.json`、`implementation_check.json`。临时研究文件不进入代码同步边界。
- 完整GPU训练/DDP运行尚未由agent验证；用户手动启动后以实际W&B结果为准。
- 2026-09-17已执行Mutagen flush成功，随后四个session（src/configs/scripts/root-code）均为Watching for changes、端点已连接且无冲突。生产代码和根脚本已同步到node1，未修改远端环境或创建Git commit。
