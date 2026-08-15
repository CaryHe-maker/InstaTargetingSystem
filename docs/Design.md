# 系统设计

InstaTargetingSystem 是面向 ERP 全景视频的单目标跟踪系统。球面几何、HiT-Small、深度辅助证据和事务式控制器通过稳定数据契约组合为 RGB-only 与 RGB-D 两条运行线路。

## 架构边界

```text
FrameSource
    -> FramePacket
    -> DepthAwareTrackController.plan
    -> SphericalGeometry.cropViews (120° ViewSpec)
    -> TrackerBackend.infer
    -> SphericalGeometry.localBoxToBfov
    -> StateEvaluator (local boxes + FuseBox)
    -> TrackStateMachine
    -> TrackResult
    -> ResultSink
```

| 模块 | 职责 |
|---|---|
| `core` | 配置、错误类型、不可变数据对象和结构化协议 |
| `data` / `io` | 视频、图像、AirSim360、深度与结果文件读写 |
| `geometry` | ERP、球面 BFoV 和局部透视视图转换，以及 seam-aware 框几何 |
| `tracker` | HiT-Small 会话、模板管理、深度编码和分数融合 |
| `controller` | 多帧预测、固定视域规划、候选融合、状态转换和模板策略 |
| `app` | 运行时组装、逐帧调度、比赛入口和 AirSim360 CLI |
| `visualization` | 中间视图与最终结果的诊断图像写入 |

## 固定视域搜索

每帧先由多帧运动预测给出 `c1`。普通 ROUND_1 使用以 `c1` 为中心的四角 120° 视域；ROUND_2 使用 ROUND_1 最高置信度候选中心。UNCERTAIN/RECOVERING 的 ROUND_3 和 LOST 的 ROUND_1 使用固定 front/right/back/left/up/down 六面 cubemap。所有视域均不缩放。

相邻四角视域中心距离为 `80°`，即相对 seed 的局部偏移 `±40°`，从而获得 `1/3` 的相邻视域覆盖和 `1/9` 的共同中心覆盖。`LOST` 不在第一轮围绕 `c1` 取四角视域，而是先做六面全景搜索。

## 候选融合和证据

`StateEvaluator` 保留每个原始局部框，并按重合率从高到低做确定性一对一配对。每个 FuseBox 最多融合两个不同视域的框；融合框是 seam-aware 最小联合框，显式标记 `fused=True`。融合置信度为：

```text
1 - ((2 - b - a) * (1 - y) / 2)
```

第一轮普通状态的融合生成阈值是 `firstRoundFusionOverlap=0.30`；LOST 第一轮和所有后续轮次使用 `OverlapThreshold=0.70`。FuseBox 只有在融合重合率超过 `OverlapThreshold`、融合置信度超过 `SuccessRate` 且两个源框均达到 `FusionSourceMinConfidence=0.80` 时，才是 `RELIABLE_FUSED`。最终轮次即使候选较弱也必须输出最高候选；无候选才使用运动 fallback。

## 状态和事务

状态机消费四级证据：`RELIABLE_FUSED`、`RELIABLE_SINGLE`、`WEAK`、`MISSING`。`TRACKING` 的弱/缺失进入 `UNCERTAIN`；`UNCERTAIN` 可靠时回到 `TRACKING`，缺失或 patience 耗尽时进入 `LOST`；`LOST` 可靠 FuseBox 直接回到 `TRACKING`，可靠单框进入 `RECOVERING`；`RECOVERING` 可靠 FuseBox 立即找回，可靠单框需连续 `recoverConfirmFrames` 帧确认。

控制器以 `stateRevision`、`transactionId`、`attemptIndex`、模板修订号和 `viewId` 校验响应。中间轮次只保留帧内候选，不修改跨帧状态；每帧最终只调用一次状态转换、提交一次结果。可靠观测才允许更新运动历史和模板。

## 输出可靠性

开发和比赛结果均按帧序校验。写入器先写 `.partial` 文件，仅在结果数量与预期帧数一致时发布最终文件。模型、配置、解码、几何和输出错误通过项目异常类型传播到 CLI 退出码。
