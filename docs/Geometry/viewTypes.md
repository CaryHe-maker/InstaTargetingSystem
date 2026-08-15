# 视域与框的种类

## ERP 全景帧

ERP 把 yaw 线性映射到图像 x，把 pitch 线性映射到 y。水平轴循环，左右边界是同一条经线；垂直轴不循环，顶部/底部对应球面极点。FramePacket.rgb 和结果 bbox 都使用这套像素空间。

## BFoV

BFoV 是球面相机视域：中心 SphericalPoint、水平/垂直 FOV 和 roll。它既用于定义 LocalView 相机，也用于表达目标在球面上的中心与角尺寸。FOV 必须小于 180 度，才能用透视相机模型。

## ViewSpec

ViewSpec 在 BFoV 之外增加 viewId 和局部输出宽高，是一次实际裁剪请求。当前搜索视图固定 120×120 度、256×256 像素；模板初始化视图同样通过 ViewSpec 表达。

## LocalView

LocalView 是 BFoV 对应的透视平面采样结果，包含 RGB、可选深度和原 ViewSpec。HiT 的 bbox 只在这个平面内有意义。

## ERP bbox 与局部 bbox

二者都使用 BBoxXYWH，但语义由上下文决定。ERP bbox 允许 x+width 超过图宽表示跨缝；局部 bbox 不循环，必须被裁剪到 LocalView 边界。文档和新接口应明确写出所在空间，避免直接混用。

## 搜索视域组合

四角视域围绕预测中心形成局部覆盖，六面 cubemap 覆盖整球。具体中心布局和状态路线见 `Controller/viewPlanning.md`。

