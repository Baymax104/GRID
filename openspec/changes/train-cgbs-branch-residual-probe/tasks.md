## 1. 协议
- [x] 1.1 创建三门槛顺序计划与第一门槛设计规格
## 2. 实现
- [x] 2.1 实现有界training缓存、分片writer、严格loader与datamodule
- [x] 2.2 实现匹配残差头、frontier目标、冻结来源与能量beam
- [x] 2.3 装配cache/train/inference配置与根脚本
## 3. 验证与交付
- [x] 3.1 覆盖冻结、off/零起点、mask、来源、缓存损坏、候选去重及评分语义测试
- [x] 3.2 验证Hydra、真实Bash参数、dry-run与OpenSpec strict
- [x] 3.3 更新研究计划与执行文档，检查同步，提供人工启动命令
## 4. 等价吞吐修复
- [x] 4.1 共享历史投影及global attention，跨候选和层复用，保留branch分块内存边界
- [x] 4.2 对照旧公式验证两臂非零输出、loss、参数梯度、mask、分块与跨更新；验证投影次数
- [x] 4.3 运行聚焦回归与OpenSpec strict，记录交付与尚未实测的GPU吞吐
