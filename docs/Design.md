# 系统设计

InstaTargetingSystem 是面向 ERP 全景视频的单目标跟踪系统。系统将球面几何、真实 HiT-Small 局部跟踪、深度辅助证据和事务式控制器分离，通过稳定的数据契约组合为 RGB-only 与 RGB-D 两条运行线路。

## 架构边界

```text
FrameSource
    -> FramePacket
    -> DepthAwareTrackController.plan
    -> SphericalGeometry.cropViews
    -> TrackerBackend.infer
    -> SphericalGeometry.localBoxToBfov
    -> DepthAwareTrackController.consume
    -> TrackResult
    -> ResultSink
```

模块职责如下：

| 模块 | 职责 |
|---|---|
| `core` | 配置、错误类型、不可变数据对象和结构化协议 |
| `data` / `io` | 视频、图像、AirSim360、深度与结果文件读写 |
| `geometry` | ERP、球面 BFoV 和局部透视视图之间的转换 |
| `tracker` | HiT-Small 会话、模板管理、深度编码和分数融合 |
| `controller` | 运动预测、候选评估、状态转换、恢复搜索和模板策略 |
| `app` | 运行时组装、逐帧调度、比赛入口和 AirSim360 CLI |
| `visualization` | 中间视图与最终结果的诊断图像写入 |

## 同步执行模型

`runTracking()` 在调用线程中顺序执行。每帧完成读取、规划、裁剪、推理、回投影、控制器提交和结果写入后，才读取下一帧。控制器可在同一帧返回 `MoreViewsRequired`，驱动在配置预算内执行第二次搜索；无论搜索次数如何，每帧只提交一个 `TrackResult`。

配置中的 `runtime.*QueueCapacity` 字段由严格配置加载器校验，用于保持配置模式兼容性；同步执行函数不创建工作线程或运行时队列。

## 模型线路

`buildRuntime()` 默认使用 `PyTorchHiTSession`：

- RGB-only：一个 HiT-Small 会话。
- RGB-D：一个 RGB HiT-Small 会话和一个深度伪彩色 HiT-Small 会话。

两个 RGB-D 会话拥有独立模型与模板状态。控制器不直接访问模型设备资源，所有 CUDA 生命周期由后端负责。

## 控制器一致性

控制器是跟踪状态的唯一写入者。搜索计划携带 `stateRevision`、`transactionId`、`attemptIndex` 和模板修订号，过期或乱序响应会被拒绝。状态机覆盖 `TRACKING`、`UNCERTAIN`、`RECOVERING` 和 `LOST` 公共状态，并使用运动、尺度、深度、多视角支持度和一致性证据决定观测接纳与恢复搜索。

## 输出可靠性

开发结果和比赛结果均按帧序校验。写入器先写 `.partial` 文件，仅在结果数量与预期帧数一致时发布最终文件。模型、配置、解码、几何和输出错误通过项目异常类型传播到 CLI 退出码。
