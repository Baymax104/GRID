# LETTER 前缀约束性能修复（2026-10-09）

> 后续更新：用户已明确授权重新启动六组正式训练，新 run 均通过启动核验。见 [修复后启动记录](letter-launch-20261009.md)。本文“保持停止”描述的是修复核验结束时的阶段状态。

## 实现

在 `src/recommendation/letter/backbone.py` 中加入 `LetterPrefixLogitsProcessor`，每个解码步对全部 beam 前缀执行一次 `tolist()`，使用现有 CPU trie 查合法词，合并行列索引后批量写入加性 mask。原先每个 beam 的回调与索引写入产生大量 CUDA 同步。

`generate()` 通过 HF `logits_processor` 接入；保留原 `allowed_tokens()` 作为兼容接口与差分参照。继续使用 HF T5 beam search、beam20/top10、四码加 EOS、完整词表概率、length_penalty=1、early_stopping、全目录候选及原排序协议。state_dict、checkpoint identity、训练 loss 和实验配置不变；没有复用其他项目模型。

OpenSpec：`openspec/changes/optimize-letter-prefix-constraints/`。

## 本地验证

```powershell
uv run --no-sync pytest tests/recommendation/test_letter_backbone.py tests/recommendation/test_letter_module.py tests/data/components/test_letter.py tests/test_letter_config_script.py tests/test_letter_issue_commands.py -q
openspec validate optimize-letter-prefix-constraints --strict
```

43 项测试通过。新增测试涵盖每步只读取一次矩阵、所有目录深度及 EOS/padding mask 对照、空候选与非法前缀失败、三个随机种子与全 logits 同分边界的 HF 完整生成对照。原 checkpoint state_dict 严格加载不变。

## node1 有限核验

使用 `mutagen_sync.ps1 flush` 同步，三个 session 均 `Watching for changes`，无 conflict。本地与 node1 backbone 文件 SHA256 相同：`ca54930ef8739baf9b770910dfb8b5c86d32676b3cae62f66cffb05f42daabd4`。

物理 GPU4 对应进程内 CUDA0，A100 80GB，开始及结束 GPU4 无其他 GPU 分配；没有隔离其他进程的独占保证。Torch 2.9.1+cu128、Transformers 5.14.1、FP32/medium、CPU threads=4。直接在有限诊断进程中加载保留的推荐 checkpoint，不执行 Trainer、不创建 W&B run、不上传 Artifact。

六个 checkpoint 均检查 catalog SHA256、`on_load_checkpoint` identity 及 `load_state_dict(strict=True)`。按 evaluation 文件路径排序读取 TFRecord，使用正式 `LetterPreprocessor(training=False)`，每组取前 67 个用户，组成 32、32、3 三个 batch。第一 batch 预热后交替原 HF 回调与生产批量处理器三次，另外两个 batch 各比较一次。18 个不同 batch、402 个用户、30 次前后输出比较，所有 IDs 和 scores 均 `torch.equal`，输出合法唯一且分数有限。

| Issue / 数据 / seed | checkpoint step | 原回调中位秒 | 修复中位秒 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| BMX-60 / Beauty / 42 | 18500 | 0.3335 | 0.0670 | 4.98× |
| BMX-61 / Beauty / 200 | 17500 | 0.3278 | 0.0678 | 4.83× |
| BMX-62 / Beauty / 2026 | 18500 | 0.3291 | 0.0679 | 4.85× |
| BMX-63 / Sports / 42 | 10500 | 0.3321 | 0.0713 | 4.65× |
| BMX-64 / Sports / 200 | 10500 | 0.3250 | 0.0661 | 4.91× |
| BMX-65 / Sports / 2026 | 11000 | 0.3299 | 0.0684 | 4.82× |

中位耗时仅对应第一 batch 的三次重复，不是全量 validation 时间。原始证据位于本地与 node1 的 `logs/letter-prefix-fix-20261009/verification.json`，包含 checkpoint 路径/哈希、catalog 指纹、输入哈希、用户 ID、软件版本及逐次耗时。此前共享 GPU0 的约 40 倍原型测量与本次环境不同，不能直接混用。

## 状态与边界

修复已应用并同步，有限诊断进程已结束，六组 LETTER 正式训练保持停止。本次不构成全量 DDP 验证、正式 Testing 或基线有效性验收。保留的旧 checkpoint 仅用于差分诊断；恢复实验需要新的 run/source 记录及明确启动指令。

forward 中重复计算 CE 的小冗余未单独测得影响，本次只修复已证实的前缀约束慢路径。
