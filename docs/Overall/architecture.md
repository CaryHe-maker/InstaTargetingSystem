# 系统架构与数据流

## 一帧经过哪些表示

一帧进入系统后不会直接送给 HiT。它依次经过四种空间：

1. `FramePacket` 保存 ERP 全景 RGB，以及可选分割数据。
2. Controller 产生若干 `ViewSpec`，每个 ViewSpec 描述球面中心、水平/垂直 FOV 和输出尺寸。
3. Geometry 将 ViewSpec 裁成 `LocalView`，HiT 在局部透视平面中产生 `LocalObservation`。
4. Geometry 将局部框边界一次投影到球面，同时拟合紧致 BFoV 和直接 ERP bbox；Runtime 再附加外观概率、运动概率、SingleScore 和投影质量，形成 `ProjectedObservation`。
5. Controller 只在统一的 ERP/球面语义中比较候选，并且只使用 `singleScore` 排序和融合；三种状态的最佳融合框都以上一可信框面积执行同一套自适应裁剪。

最终的 `TrackResult` 同时携带 ERP `bbox` 和球面 `bfov`，使本地评估与比赛提交共用同一次跟踪结果。

## 依赖方向

Core 位于最底层。Geometry、Tracker、Controller、Data 依赖 Core，但彼此通过协议和值对象通信。Runtime 是组合根，可以依赖所有运行模块；Visualization 和 Evaluation 只能旁路读取结果，不应反向影响 Controller。

这种边界有两个直接目的：

- 更换 HiT 会话实现时，不需要修改状态机和球面几何。
- 更换数据源或结果格式时，不需要修改候选融合算法。

## RGB-only 主运行路线

Runtime 建立一个 HiT 会话，局部 RGB 直接得到模型框与外观分数。Tracker 把同一轮的 RGB 视图组成一个 tensor batch，Runtime 再将局部框回投为 `ProjectedObservation` 交给 Controller。比赛视频、本地 AirSim360 与通用图像序列共用这条线路。

## 关键实现路径

- 数据契约：`src/instatarget/core/types.py`
- 协议边界：`src/instatarget/core/protocols.py`
- 组合根：`src/instatarget/app/driver.py::buildRuntime`
- 顺序驱动：`src/instatarget/app/driver.py::runTracking`
- 状态所有权：`src/instatarget/controller/track_controller.py`
- 局部推理：`src/instatarget/tracker/backend.py`
- 球面投影：`src/instatarget/geometry/spherical_geometry.py`
- 分数校准与合成：`src/instatarget/controller/fused_score.py`

## 修改架构时的约束

新增模块应尽量消费已有值对象。若必须增加跨模块字段，应先修改 Core 类型和协议，再修改生产者、消费者和协议测试。不要让 Tracker 直接修改状态，也不要让 Visualization 参与分数计算，否则同一算法在关闭可视化后可能产生不同结果。

