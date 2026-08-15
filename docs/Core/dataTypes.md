# 核心数据类型

## 图像与空间类型

`FramePacket` 是一帧的唯一跨模块容器，包含序列 ID、帧号、单调时间戳、ERP RGB、可选 `DepthPlane` 和可选 `SegmentationPlane`。RGB、深度和掩码必须空间对齐。

`SphericalPoint` 同时保存 yaw/pitch 和三维单位向量。单位向量用于跨经线和极点附近的稳定计算；yaw/pitch 用于配置、显示和文件格式。

`BFoV` 用球面中心、水平 FOV、垂直 FOV 和 roll 描述球面视域。`BBoxXYWH` 是 ERP 或局部平面的像素框，必须结合所在图像宽高解释。

## 查询链类型

- `ViewSpec`：请求裁剪什么球面方向，以及输出局部图尺寸。
- `LocalView`：Geometry 实际裁出的 RGB、可选深度和原始 ViewSpec。
- `LocalObservation`：Tracker 返回的局部 bbox、模型/外观/深度/融合分数。
- `ProjectedObservation`：局部框回投 ERP 后的 bbox/BFoV，并带有运动、尺度和深度证据。

不要跳过 ProjectedObservation 直接比较不同 LocalView 中的 bbox，因为它们的相机中心不同，局部像素坐标没有可比性。

## 控制与结果类型

`SearchPlan` 绑定帧身份、状态 revision、事务 ID、attemptIndex、视图和模板命令。`MoreViewsRequired` 表示同帧继续查询；`FrameCommitted` 表示该帧已经原子提交。

`TrackResult` 是对外结果，包含 ERP bbox、BFoV、置信度、公开状态、valid、深度摘要和结果来源。`valid=False` 不等于没有 bbox；它可能携带运动预测框，但该框不得进入可靠测量历史。

## 修改原则

类型构造函数承担范围和形状校验。新增字段应明确单位、坐标系、可空语义和谁拥有写权限，并同步 `tests/unit/test_core_types.py`。

