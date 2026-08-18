# 逐帧跟踪循环

## 初始化阶段

第 0 帧不是普通搜索帧。Controller 根据给定初始化框生成模板 ViewSpec，Geometry 裁剪后由 backend 编码模板，随后 Controller 初始化运动历史和状态。初始化结果置信度为 1，来源为 INITIAL。

## 普通帧算法

`runTracking()` 对每帧只调用一次 source.read，然后执行一个内部多轮循环：

1. `beginFrame()` 产生当前 round 的 SearchPlan。
2. `cropViews()` 一次裁剪本轮所有视图。
3. `backend.infer()` 按计划顺序把本轮全部视图交给 backend。PyTorch HiT 会话执行一次 RGB tensor batch；RGB-D 路径随后对有深度的视图执行一次 depth tensor batch。每个 attempt 的 expectedRevision 严格递增，但第二轮 KEEP 保持模板特征内容不变。
4. `calibrateLocalAppearanceProbabilities()` 把 backend 原始 `fusedScore` 校准为独立的 `appearanceProbability`，不覆盖原分；旧 `remapLocalObservationFusedScores()` 仅是兼容别名。
5. `_projectObservation()` 将局部框边界一次回投，直接生成 ERP bbox、紧致 BFoV、边界与膨胀诊断。
6. 每个局部图中心与该帧预测中心的大圆夹角生成同帧视图运动分数，再按 70/30 合成 `singleScore`；检测框在局部图内部的位置不改变该视图的运动先验。
7. `controller.consume()` 在 TRACKING/UNCERTAIN 第一轮先用 Fusor 选择最佳中心，再围绕该中心请求第二轮 VStype1 四角 4 张视图；无候选时以预测中心回退。第二轮结束时把两轮观测统一交给 Fusor，完成单框排序和两框融合后提交。当前正常线程不会进入 LOST，低于 LT 或全零缺失仍提交 UNCERTAIN；显式 LOST 组件保留 6 张 cubemap 加 4 张 Type1 的单轮 Fusor 路径。三种模式的最佳融合框都以上一可信框面积执行同一套参考面积裁剪；当参考面积不小于最小合并框时直接返回合并框，不再放大。

中间轮的视图、局部观测和投影观测暂存在 `visualizationBatches`，但只在处理区间结束后写图。Controller 另外在 FrameTransaction 中保存各轮投影观测；第一轮局部图不会重复执行 backend 推理，但其投影观测会在第二轮提交时参与 Fusor 排序与融合。

## 输出顺序

Controller 先完成原子提交，Runtime 再调用 sink 和 result recorder。因此 sink 失败不会让同一帧重新进入 Controller；调用方应将输出错误视为序列失败，而不是重试 consume。

## 复杂度

裁剪、预处理、投影和后处理仍近似随视图数线性增长，但 HiT 模型调用已按轮批处理，不再是每张图一次 forward。正常线程的 TRACKING 与 UNCERTAIN 都是 4+4 张、两个 batch；显式 LOST 组件是 10 张、一个 batch。RGB-D 每轮还会追加一个 depth batch。降低总延迟时应同时统计状态/round 分布、batch size、模型 forward 数、GPU 利用率和峰值显存。

