# 控制器模块

`DepthAwareTrackController` 是控制状态的唯一写入者，负责将球面运动预测、视图规划、候选评估、状态机和模板策略组合为逐帧事务。

## 主要接口

- `buildInitialization(frame, initialBox)`：为第 0 帧生成模板视图和模板框。
- `commitInitialization(plan, depthSummary)`：建立运动历史并提交初始化结果。
- `beginFrame(frame)` / `plan(frame)`：预测目标并生成带事务标识的搜索计划。
- `consume(plan, observations)`：提交一次尝试，或返回 `MoreViewsRequired`。
- `update(plan, observations)`：兼容调用路径，强制提交当前尝试，不启动额外搜索。

## 事务一致性

控制器验证序列 ID、帧索引、事务编号、尝试编号和模板修订号。重复、乱序或跨帧响应都会触发 `ProtocolError`。同一帧最多执行 `tracking.maxAttemptsPerFrame` 次尝试，并受 `maxViewsPerFrameTotal` 约束。

## 决策依据

`StateEvaluator` 对局部观测进行球面聚类，计算后端、运动、尺度、深度一致性、支持度和多视角一致性。`TrackStateMachine` 根据评估结果和连续弱观测次数选择跟踪、犹豫、恢复或丢失状态。`RecoveryPlanner` 生成环形、立方体和全局扫描视图；`TemplatePolicy` 只在稳定条件满足时更新近期或稳定模板。

## 深度行为

深度摘要仅在 `depth.enabled` 且帧提供有效深度时计算。RGB-only 配置的 `depthProcessor` 和 `depthEncoder` 均为空，控制器仍保留统一的可选深度字段，不会因缺少深度而改变接口形状。
