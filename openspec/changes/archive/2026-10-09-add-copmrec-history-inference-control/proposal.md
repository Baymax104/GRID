# 固定 CoPMRec 检查点的最终历史排除对照

## Why

用户授权一次Beauty/seed42单卡推理，以分离最终商品排除的作用。已有M2关闭历史侧残差但保留最终排除，不能回答该问题。

## What Changes

- 新增仅推理的正式Full检查点兼容子类，保持encode/dense logits/稳定排序，只省略最终历史掩码。
- 新增薄experiment和根脚本，通过统一src.main运行。
- 保留CPU参数/原始评分一致性测试，记录精确命令和独立产物核验。

## Impact

不改变正式CoPMRec、原LIGER、训练或checkpoint选择；新增0训练、0Validation、1Testing，内部实证独立登记。
