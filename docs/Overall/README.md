# InstaTargetingSystem 文档索引

当前实现文档按模块组织。每个 `structure.md` 只说明边界和阅读路线，算法细节放在同目录专题文档中。
除 `docs/Prepare/` 外，本文档树按当前代码维护；Prepare 是历史设计材料，不作为当前实现规范。

| 模块 | 结构入口 | 主要专题 |
|---|---|---|
| Overall | [structure.md](structure.md) | [架构](architecture.md)、[完整运行线程](runtimeThread.md)、[运动评分与边界回投](motionProjectionUpgrade.md)、[优化入口](optimizationMap.md) |
| Core | [../Core/structure.md](../Core/structure.md) | 数据类型、事务协议、严格配置 |
| Runtime | [../Runtime/structure.md](../Runtime/structure.md) | 组件装配、逐帧循环、计时生命周期 |
| Controller | [../Controller/structure.md](../Controller/structure.md) | 状态机、运动预测、视图规划、StateEvaluator、校准、模板事务 |
| Tracker | [../Tracker/structure.md](../Tracker/structure.md) | RGB HiT、批量推理、模板缓存 |
| Geometry | [../Geometry/structure.md](../Geometry/structure.md) | 视域类型、坐标转换、透视投影、跨缝 |
| Data | [../Data/structure.md](../Data/structure.md) | AirSim360、帧源、伪真值、结果 sink |
| Visualization | [../Visualization/structure.md](../Visualization/structure.md) | 阶段产物、最终绘制、实例 ID、处理计时 |
| Evaluation | [../Evaluation/structure.md](../Evaluation/structure.md) | 平面/球面指标、性能统计、验证流程 |
| Competition | [../Competition/structure.md](../Competition/structure.md) | 序列运行、结果格式、容器环境 |
| Training | [../Training/structure.md](../Training/structure.md) | 样本生成、训练边界、校准数据流程 |

第一次阅读建议按以下顺序：完整运行线程 → Controller 视图规划 → StateEvaluator → 状态机 → Geometry 坐标转换 → Tracker HiT。

