## Context

用户确认基础 CoPMRec 是已有训练/推理证据的基础版本，最终收益转化仍待完成。旧补强通过 root shell 和 Hydra experiment 公开启动，四份配置/脚本测试依赖这些入口，算法测试不依赖它们。清理授权仅覆盖旧尝试入口和失效的启动契约。

## Goals / Non-Goals

**Goals:**

- 退役 19 个可启动入口，归档原始文件字节、哈希及 4 份旧启动契约测试。
- 保持基础 CoPMRec/LIGER 的训练和推理配置可 compose，保留旧算法实现与纯内存测试。
- 删除通过现有 Mutagen 三会话同步，记录已完成及尚未验证的边界。

**Non-Goals:**

- 不删除 checkpoint、数据、运行日志、W&B 产物或旧实验结果；不停止训练进程。
- 不实现新收益转化组件，不启动正式训练/推理，不修改 baseline。

## Decisions

1. 用 `docs/archive/copmrec-conversion-entrypoints-20261003/retired-files.zip` 和 manifest 保存实际字节；归档先完成 CRC/逐文件 SHA 验证，再逐文件删除。清单中的源和目标 resolve 后须属于仓库，拒绝覆盖已有归档。
2. 仅删除根脚本和 experiment group 配置。component config、算法和 writer 保留，便于理解旧 checkpoint/trace 及单元复现；它们不代表仍推荐使用旧实验。
3. 四份只服务旧入口的测试随原始文件归档，新测试核验根脚本退出、Hydra 拒绝退役 experiment、归档完整及基础配置未变。保留的基础脚本测试继续检查语法、quoting、错误输入和额外 override。
4. manifest 记录基础入口和相关核心文件的清理前 SHA，以检查无意改动；不以远端 Git 判断版本。
5. 研究文档修正上一轮“退回基础方法即可完成”的解释，明确新组件须把效果落实到最终 Recall/NDCG；设计输出归 research，清理技术记录归 GRID。

## Risks / Trade-offs

- 旧复制命令失效 → 当前计划及交付文档增加统一退役提示，原始命令仅供追溯。
- 算法/组件仍可被手工装配 → 明确只清理标准入口，保留技术参考不授权恢复旧阶段。
- 同步传播删除 → 只删除已验证归档清单，沿用 mutagen.yml 保护边界，flush/status 后检查远端入口缺失与基础入口存在。
- 已有运行可能加载旧代码 → 本轮不操作运行进程，删除入口不等于停止已启动实验。

## Migration Plan

归档并验证，退出入口，替换旧启动测试，执行聚焦检查，完成 Mutagen flush/status，核对远端路径。需要历史复现时从 ZIP 在隔离目录读取原始文件并另行明确协议，不自动解压回标准入口。
