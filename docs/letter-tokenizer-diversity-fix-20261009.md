# LETTER Tokenizer Diversity 采样修复（2026-10-09）

## 实现

`src/quantization/letter/tokenizer.py::diversity_loss` 的默认采样分支现在分别一次性读取 labels 和 ids，按 codebook 索引递增顺序建立 CPU group 成员列表。继续按原样过滤 self，并逐样本调用 Python random.choice。原检查、loss、显式 positives 分支、state_dict 和 checkpoint identity 保持不变。

这消除了原实现 batch1024、四层4136次 CUDA 标量转整数。不增加 group 缓存，不需要额外的 update_groups/checkpoint load 缓存维护。20000 epoch、每 epoch 分组、n_jobs10、Sinkhorn50轮、loss 权重和精度配置均未修改。

OpenSpec：`openspec/changes/optimize-letter-tokenizer-diversity-sampling/`。

## 本地验证

```powershell
uv run --no-sync pytest tests/quantization/test_letter_tokenizer.py tests/quantization/test_letter_tokenizer_module.py tests/data/components/test_letter.py tests/test_letter_config_script.py tests/test_letter_issue_commands.py -q
uv run --no-sync ruff check src/quantization/letter/tokenizer.py tests/quantization/test_letter_tokenizer.py
openspec validate optimize-letter-tokenizer-diversity-sampling --strict
```

36项聚焦测试通过，Ruff及OpenSpec strict通过。新增回归在旧实现上明确失败于 tensor 标量读取；修复后 labels/ids 各仅读取一次。测试覆盖不规则 group labels、重复 ids、三个 seed 的 loss/梯度/Python RNG 终态、显式 positives 不采样、非法 positives 和 singleton 失败，以及三步 AdamW 的参数与 optimizer 状态逐位等价。

## node1 有限真实 checkpoint 核验

经 `mutagen_sync.ps1 flush` 同步，三个 session 均 Watching for changes，无 conflict。本地与 node1 tokenizer SHA256相同：`aa8a4d0aaae9b086b53c3716dead27e20293d6249b28f4d64c94c8a18f59bf77`。

使用 Beauty/Sports × seed42/200/2026 六个现有 validation-selected best checkpoint，严格检查原 input/architecture identity 并加载完整 state_dict。实际共同内容和各自 CF 使用公共 loader 读取。

物理GPU4 → CUDA0，A100 80GB、Torch2.9.1+cu128、FP32/medium、诊断进程 CPU threads=4。每组取前1024商品，预热后分别运行原函数与生产修复函数，三个固定随机种子下交替计时 forward+backward，不启用 profiler或计数 hook、不执行 optimizer 更新。所有返回 tensor、参数梯度及 Python RNG 终态均逐位相同。

| Dataset / seed | best step | 原实现中位ms | 修复中位ms | 加速 |
| --- | ---: | ---: | ---: | ---: |
| Beauty / 42 | 24000 | 93.25 | 25.33 | 3.68× |
| Beauty / 200 | 48000 | 99.11 | 23.85 | 4.16× |
| Beauty / 2026 | 72000 | 100.31 | 28.99 | 3.46× |
| Sports / 42 | 72000 | 99.31 | 23.27 | 4.27× |
| Sports / 200 | 36000 | 97.50 | 24.27 | 4.02× |
| Sports / 2026 | 36000 | 103.50 | 24.85 | 4.17× |

另对每组实际尾 batch（Beauty837、Sports949）严格加载旧 checkpoint 的 AdamW 状态，并分别使用原函数/修复函数继续两步。两条路径的每步输出、梯度、Python RNG、更新后完整 model state_dict、optimizer state及param_groups均相同。随后单独计数生产路径的 CUDA tensor转整数，六组均为0；其他固定错误检查仍可能同步，不声称消除了所有CUDA同步。

首组验证成功保存后，诊断脚本曾在清理临时变量时出现 NameError；已修正并从已保存记录继续其余五组，最终六组全部通过。该错误不在生产模型或训练入口中。

原始证据：本地与 node1 `logs/letter-tokenizer-diversity-fix-20261009/verification.json`。包含 checkpoint路径/哈希、step、输入哈希、逐次耗时、尾batch大小、等价性和optimizer核验结果。前一轮原型5.25倍测量的线程/负载及样本范围与本轮不同，不能替代本轮生产修复测量。

## 状态与边界

修复已应用并同步，有限验证进程已结束，没有启动或重启正式实验、没有发布新W&B run/Artifact、没有保存新checkpoint。仅独立验证进程执行有限的成对optimizer更新。

该结果覆盖六个Beauty/Sports真实checkpoint与完整/尾batch，不代表全量训练加速、完整epoch轨迹或Toys真实checkpoint验收。Toys正式上游尚未生成，未来运行会使用同步后的实现；运行中的旧进程不会因源码同步自动加载该修复。
