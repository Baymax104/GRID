## Why

成熟 C 商品评分及原型聚合诊断未支持旧路线，尚未检验可学习商品空间。用户授权继续第一关，实现固定/可学习 bank 的独立内容序列训练对照。

## What Changes

- 增加独立商品级 Transformer 和共享低秩 adapter，两臂同初始化与数据顺序。
- 训练缓存去重、按用户隔离校准数据；固定2000步终点。
- 提供统一入口训练/配对评价、证据完整性检查、启动脚本和CPU测试。
- 不实现后两关，不自动训练、不改变 A 或 SID。

## Capabilities

### New Capabilities
- `content-representation-qualification`: 第一关表示资格实验。

## Impact

新增推荐模型、data组件、配置、writer和脚本；复用共享Artifact、checkpoint、analysis协议。
