# SASRec 正式输入与运行准备核验（2026-09-30）

## 结论与范围

三个数据集正式目录已固定到生产run所发布的v0；SID bundle、内容embedding bundle与原始items目录的key集合完全相同。九个dataset/seed单元通过node1 CUDA0的统一入口训练dry-run，默认4 workers、batch128，每次仅1次optimizer update且exit0。

本次完成可运行准备，不是正式准确性实验。dry-run自动禁用validation/testing、checkpoint/结果writer及W&B logger，不登记正式run或metrics。正式训练后才能填写validation-selected best并开展完整testing。多卡与50000步收敛仍未实测。

## 输入来源与全量核验

| 数据集 | SID Artifact | 商品数 | evaluation/testing用户数 | 内容 Artifact |
|---|---|---:|---:|---|
| beauty | rkmeans_inference_beauty-semantic-id:v0 | 12101 | 22363 | sem_embeds_inference-semantic-embedding:v5 |
| sports | rkmeans_inference_sports-semantic-id:v0 | 18357 | 35598 | sem_embeds_inference-semantic-embedding:v6 |
| toys | rkmeans_inference_toys-semantic-id:v0 | 11924 | 19412 | sem_embeds_inference-semantic-embedding:v7 |

通过W&B SDK的run.logged_artifacts获取实际版本/digest/文件信息；使用共享load_model_output读取bundle内容，不以名称推断集合。全量扫描node1 data/{dataset}/{items,training,evaluation,testing}，对每个压缩TFRecord分片计算SHA256；manifest按文件名排序串接 filename、零字节、文件SHA256及换行再hash。用户key hash为排序后的little-endian int64数组SHA256。

所有split商品均属于catalog；items集合恰等于catalog；evaluation/testing均无空历史、无重复用户且用户集合相同。三个数据集training用户数也与evaluation/testing一致。内容embedding仅核验keys，SASRec不使用其向量值。未计算任何新的完整testing模型指标。

### beauty

- 固定 URI：`wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&alias=v0&file=merged_predictions_tensor.pt`。
- Artifact digest：`19ace08283163fbe687287fbacaa3842`。
- bundle 文件 SHA256：`c95072fc46b7a6a9359fda3c1225625cb2c3625a3be8b7683652ada17949489b`。
- catalog SHA256：`5ff9b2543331cb866347349d3d66026bcfe1a5a4f501716ab09c3832ea06b43c`。
- training manifest：`a5cb8337740c12d579aec84fb99cffdf5dc9418d5796d71605139163a42c6c9b`。
- evaluation manifest：`3eef01628ef02d9421482fb42602945df5cecdba18521a2bdc494c2530a11c71`。
- testing manifest：`f2ca1d268ea57618ba3013414536fe9b6c4bd1ad990dda1b856f21ec40a03486`。
- evaluation/testing 用户 key SHA256：`5c14775a3aec5907a79e94c4cc0b2a00be6344c0e72cb2e40cca75e9c34744d9`。

### sports

- 固定 URI：`wandb://baymaxam/GRID/jcyr5l3p?role=semantic_id&alias=v0&file=merged_predictions_tensor.pt`。
- Artifact digest：`02b8a72342025c145025f8fe7f406f63`。
- bundle 文件 SHA256：`12f4b8bd28056f22774403a56770d43801f023ba0eeec5f6b1be577b7bb4b80b`。
- catalog SHA256：`a55feb8adf0d40324cac9148910870254b91411a418361ccd8d1e06be39b91b1`。
- training manifest：`c1ac70c51374a64fc7d03dc2ecfd2e7350ef9257ee0c33893895f4c08b3f6e01`。
- evaluation manifest：`412429dd3688a40bedd4d285dcb19cd999be800f3ec2267a671b51c1dd73efe3`。
- testing manifest：`ce971060267672195714c58d2cf597cd285cc1e5c195868e943e75722a0016b2`。
- evaluation/testing 用户 key SHA256：`184ff09a04cda8e28549e85dd409f7339c9e2fc742821f7bc85feeaff9e15efa`。

### toys

- 固定 URI：`wandb://baymaxam/GRID/3dycz43g?role=semantic_id&alias=v0&file=merged_predictions_tensor.pt`。
- Artifact digest：`0cda6f403fd105b33b831655d06b5f0c`。
- bundle 文件 SHA256：`bc75dda4e1a5fe9542f535329aad2a33545456965c4e2e4a0157f623b9e8d5b6`。
- catalog SHA256：`a5854658cccdb1121605b268c7bbe9ebc1a187e73e19890c502edf43ee1b079a`。
- training manifest：`662a8d0c7c7ddb827cde876ef77a39ba80fe7b55b48b2db5bb16019e1e2af750`。
- evaluation manifest：`ed2703bea119b20582149cac4711e3007643aebd8049acde86f3b504af2bc90e`。
- testing manifest：`bdddf4032bb30b441fcb5ce7a5a1693dd10411915b8520ae35cdd194ac4d0261`。
- evaluation/testing 用户 key SHA256：`85eb5bc0eda44e036f47b49a706c7c44b1954481b05f06d432b280921e63d4b8`。

## 运行环境与同步

工作目录node1:/data3/weizhenyu/projects/GRID；Python3.11.10、PyTorch2.9.1+cu128、Lightning2.6.5、CUDA runtime12.8、NVIDIA A100-SXM4-80GB。CUDA_VISIBLE_DEVICES=0，逻辑devices=[0]；该卡已有任务，本次仅有界顺序运行。

执行项目mutagen_sync.ps1 flush成功，三个session均Watching for changes、无conflict。22个SASRec相关源码/配置/脚本及共享W&B reader的本地/远端SHA256一致；不使用远端Git判断版本。未改动node1虚拟环境或安装依赖；非交互SSH PATH须包含/home/weizhenyu/.local/bin以找到现有uv。

实际调用root sasrec_train.sh，经uv run python -m src.main与experiment=sasrec_train。显式--dry-run交由统一launcher限制max_steps=1、limit_train_batches=1、limit_val_batches=0、limit_test_batches=0。worker默认4，训练batch128、FP32、模型官方默认参数。有效业务writer与W&B logger被禁用，lineage callback的无W&B run warning是预期现象，不能作为正式lineage成功证据。

## 九个单元

| Issue | 数据集 | seed | 进程耗时 | 日志loss（四位小数） | exit / optimizer steps |
|---|---|---:|---:|---:|---|
| BMX-69 | beauty | 42 | 38.9s | 1.3822 | 0 / 1 step |
| BMX-70 | beauty | 200 | 74.2s | 1.3894 | 0 / 1 step |
| BMX-71 | beauty | 2026 | 46.5s | 1.3855 | 0 / 1 step |
| BMX-72 | sports | 42 | 55s | 1.3855 | 0 / 1 step |
| BMX-73 | sports | 200 | 37.8s | 1.3848 | 0 / 1 step |
| BMX-74 | sports | 2026 | 29.7s | 1.3856 | 0 / 1 step |
| BMX-75 | toys | 42 | 25.9s | 1.3867 | 0 / 1 step |
| BMX-76 | toys | 200 | 27.7s | 1.3891 | 0 / 1 step |
| BMX-77 | toys | 2026 | 25.5s | 1.3875 | 0 / 1 step |

loss均有限，仅检查梯度链路，不比较数值优劣或用于选超参数。三seed共享数据集固定catalog。

## 版本锁定修复

核验发现shared select_output_artifact原先只检查用户aliases，W&B的v0并不在[latest]中，因此alias=v0无法选择已知版本。src/utils/wandb.py现对vN匹配Artifact真实version；普通alias保持原语义，错误版本明确失败且不会回退latest。增加3项回归；Artifact reader与SASRec配置/脚本共38项测试通过，Ruff与diff检查通过。实际共享reader读取v0 SID以及v5/v6/v7内容Artifact均成功。

## 手动运行与证据位置

九个Linear单元BMX-69–77包含对应固定CATALOG、seed、data_dir的独立训练命令；正式训练使用默认50000步、val/ndcg@10选best，不自动testing。testing的BEST只能在真实训练完成后填入本dataset/seed best checkpoint。此准备核验不启动完整实验。

原始证据保存在node1 logs/sasrec-readiness-20260930/：catalog-audit.json、content-catalog-audit.json、dry-runs.json，每个dataset-seed子目录的process.log与Hydra配置。本地logs同目录保留三份JSON核验副本；不上传缓存或临时验证产物到W&B。

