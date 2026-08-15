# 逐帧跟踪循环

## 初始化阶段

第 0 帧不是普通搜索帧。Controller 根据给定初始化框生成模板 ViewSpec，Geometry 裁剪后由 backend 编码模板，随后 Controller 初始化运动历史和状态。初始化结果置信度为 1，来源为 INITIAL。

## 普通帧算法

`runTracking()` 对每帧只调用一次 source.read，然后执行一个内部多轮循环：

1. `beginFrame()` 产生当前 round 的 SearchPlan。
2. `cropViews()` 一次裁剪本轮所有视图。
3. `backend.infer()` 按同一模板 revision 推理整批视图。
4. `remapLocalObservationFusedScores()` 执行 Beta Calibration。
5. 每个局部框通过 `_projectObservation()` 回投，并计算运动中心相似度与尺度分数。
6. `controller.consume()` 决定继续或提交。

中间轮的视图、局部观测和投影观测暂存在 `visualizationBatches`，但只在处理区间结束后写图。

## 输出顺序

Controller 先完成原子提交，Runtime 再调用 sink 和 result recorder。因此 sink 失败不会让同一帧重新进入 Controller；调用方应将输出错误视为序列失败，而不是重试 consume。

## 复杂度

一帧的主要计算量近似为 `视图数 × 单次 HiT 成本 + 投影成本`。状态路由的视图数上限是 TRACKING 8、UNCERTAIN 14、RECOVERING 14、LOST 10。降低总延迟时应先统计实际 round 分布。

