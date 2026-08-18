# 分状态视域规划

实现位于 `controller/recovery_planner.py`。LOST 和 UNCERTAIN 的 cubemap 仍使用
`geometry.maxFovDeg=120`，局部输出尺寸仍为 256×256。TRACKING 从初始化后的第二帧开始
使用上一帧运动预测的目标角尺寸动态规划四角视图。

## VStype1 四角

`ViewSpecType1(center, width, height)` 返回四个 `ViewSpec`，顺序固定为左上、右上、左下、右下。
其中 `width`/`height` 是上一帧预测框的水平/垂直角尺寸；每个 ViewSpec 的 FOV 分别为
`3*width` 和 `3*height`（面积是预测框的 9 倍），随后分别限制在 30° 到
`geometry.maxFovDeg=120` 之间。下限避免目标很小时视域收缩过度，上限避免超大目标跨过
透视相机背面。中心在局部相机坐标系中按各自最终 FOV 的三分之一偏移，
因此相邻视图的重合比例与原来 120° FOV、±40° 偏移相同。中心通过
forward/right/up 基向量计算，靠近极点时仍保持局部四角语义。

TRACKING 的第一轮和第二轮都调用该动态 VStype1。首个 tracking 帧使用 frame 0 初始化时
提交的标准追踪框角尺寸；后续帧使用上一帧预测角尺寸。即使运动估计器暂时没有输出尺度，
也使用上一帧已提交 BFoV，不回退到固定四角布局。UNCERTAIN 的四角第二轮和所有 LOST
路径保持旧布局。

## VStype2 旋转 Cubemap

Cubemap 的 front 面指向传入中心，其余五面由同一局部正交基生成：right、back、left、up、down。每面为 120°，因此 cubemap 会随预测中心旋转，而不是固定使用 ERP 世界坐标的 front/right/back/left/up/down。

LOST 首轮一次生成两个 cubemap：第一个以预测中心为 front，第二个以第一个 cubemap 的 right 面中心为 front。这样不需要等待第一批推理结果，也能满足“12 张视图统一交给 Fusor”的事务约束。

## Fusor 引导的第二轮

TRACKING 和 UNCERTAIN 的第一轮观测先由 Fusor 选择一个最佳候选。第二轮以该候选的 BFoV 中心为中心创建完整的 VStype1 四角 4 张视图；第一轮没有候选时使用运动预测中心。第二轮不重复第一轮的 viewId，最终提交时将两轮观测一起交给 Fusor。当前生产路径不调用保留在 `classifier.py` 中的聚类实现。

## 预算与计划身份

单帧预算最大为 12 张：TRACKING 固定 4+4=8，UNCERTAIN 固定 6+4=10，LOST 固定 12。Planner 不生成部分 cubemap 或部分四角计划；预算不足时抛出 ProtocolError。SearchPlan 的 viewId 和 viewRoles 用于验证响应顺序，第二轮 observation 只能包含该轮明确请求的 viewId。第二轮模板命令强制 KEEP，但 expectedRevision 仍按轮严格递增。
