## 1. 主链路去耦

- [x] 1.1 从 `src/utils/launcher_utils.py` 移除 restart metadata 相关 import 与恢复分支
- [x] 1.2 保留并验证 `should_retrieve_latest_ckpt_path` 的目录解析最新 checkpoint 能力
- [x] 1.3 确认 `src/train.py` 与 `src/inference.py` 的默认主链路不再感知 restart 机制

## 2. 废弃标注与脚本清理

- [x] 2.1 为 `src/utils/restart_job.py` 添加轻量 deprecated 标注，说明不属于默认主链路
- [x] 2.2 为 `src/utils/restart_job_utils.py` 添加轻量 deprecated 标注，说明仅保留作历史/手动接入用途
- [x] 2.3 从默认训练 shell 脚本中移除 `+should_skip_retry=true` 遗留参数

## 3. 验证

- [x] 3.1 做最小静态检查，确认受影响 Python 文件语法正确
- [x] 3.2 检查相关实验配置与脚本命令仍与主链路行为一致
- [x] 3.3 复核变更后默认 train / inference 路径是否满足 spec 要求
