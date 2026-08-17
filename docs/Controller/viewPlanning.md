# 分状态视域规划

实现位于 `controller/recovery_planner.py`。所有搜索 ViewSpec 的水平和垂直 FOV 都等于 `geometry.maxFovDeg=120`，局部输出尺寸仍为 256×256。

## VStype1 四角

给定中心 c，Planner 在 c 的局部相机坐标系向左上、右上、左下、右下各偏移 40°。中心通过 forward/right/up 基向量计算，靠近极点时仍保持局部四角语义。

## VStype2 旋转 Cubemap

Cubemap 的 front 面指向传入中心，其余五面由同一局部正交基生成：right、back、left、up、down。每面为 120°，因此 cubemap 会随预测中心旋转，而不是固定使用 ERP 世界坐标的 front/right/back/left/up/down。

LOST 首轮一次生成两个 cubemap：第一个以预测中心为 front，第二个以第一个 cubemap 的 right 面中心为 front。这样不需要等待第一批推理结果，也能满足“12 张视图统一交给 Fusor”的事务约束。

## Fusor 引导的第二轮

TRACKING 和 UNCERTAIN 的第一轮观测先由 Fusor 选择一个最佳候选。第二轮以该候选的 BFoV 中心为中心创建完整的 VStype1 四角 4 张视图；第一轮没有候选时使用运动预测中心。第二轮不重复第一轮的 viewId，最终提交时将两轮观测一起交给 Fusor。当前生产路径不调用保留在 `classifier.py` 中的聚类实现。

## 预算与计划身份

单帧预算最大为 12 张：TRACKING 固定 4+4=8，UNCERTAIN 固定 6+4=10，LOST 固定 12。Planner 不生成部分 cubemap 或部分四角计划；预算不足时抛出 ProtocolError。SearchPlan 的 viewId 和 viewRoles 用于验证响应顺序，第二轮 observation 只能包含该轮明确请求的 viewId。第二轮模板命令强制 KEEP，但 expectedRevision 仍按轮严格递增。
