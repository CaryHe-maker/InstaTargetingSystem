# 数据契约

项目以不可变数据对象连接输入、几何、后端、控制器和输出模块。所有数组在进入运行时边界时执行类型、形状和有限值检查。

## 帧数据

`FramePacket` 表示一个对齐帧，包含：

| 字段 | 类型与约束 |
|---|---|
| `sequenceId` | 非空序列标识 |
| `frameIndex` | 从 0 开始的非负整数，严格递增 |
| `timestampNs` | 非负纳秒时间戳 |
| `rgb` | `uint8`，形状 `[H, W, 3]`，RGB 通道顺序 |
| `depth` | 可选 `DepthPlane`，空间尺寸与 RGB 相同 |
| `segmentation` | 可选 `SegmentationPlane`，空间尺寸与 RGB 相同 |

`DepthPlane` 使用 `float32` 二维数组、布尔有效掩码和非空单位字符串。有效位置必须为有限非负值。`SegmentationPlane` 使用可选的 `int32` 语义图和实例图，并允许附带 `semantic ID -> class name` 映射。

## 坐标与区域

- `BBoxXYWH`：连续像素坐标 `(xPx, yPx, widthPx, heightPx)`，宽高为正数。
- `SphericalPoint`：单位方向向量以及弧度制经纬角。
- `BFoV`：球面中心、水平视场角、垂直视场角和滚转角。
- `ViewSpec`：局部透视视图的 BFoV、输出尺寸和唯一 `viewId`。
- `LocalView`：局部 RGB 图及可选的同步局部深度图。

ERP 水平方向按周期坐标处理。跨越左右边界的目标框不应使用普通平面区间直接裁剪，相关算法见 [modules/geometry.md](modules/geometry.md)。

## 跟踪事务

控制器以 `InitializationPlan` 完成第 0 帧初始化，并以 `SearchPlan` 描述其余帧的局部视图、模板命令、预测运动和事务编号。后端返回 `LocalObservation`，几何模块将其转换为 `ProjectedObservation`，控制器最终提交一个 `TrackResult`。

每帧最多提交一个结果。结果包含 ERP 边界框、BFoV、置信度、状态、有效标记、结果来源和可选深度摘要。

## 数据源

项目提供三类主要读取入口：

| 数据源 | 用途 |
|---|---|
| `OpenCvVideoSource` | 比赛 `.mp4` 色彩视频 |
| `AirSim360SequenceSource` | AirSim360 RGB、深度和分割序列 |
| `ImageSequenceSource` / 通用视频源 | 开发接口与测试 |

RGB-only 由配置决定后端行为，不要求输入帧主动删除深度字段。比赛视频生成的帧不包含深度；AirSim360 帧可以包含深度，而 RGB-only 运行时会忽略该模态。
