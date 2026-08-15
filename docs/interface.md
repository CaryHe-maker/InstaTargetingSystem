# 模块接口

项目模块通过 `src/instatarget/core/protocols.py` 中的结构化协议连接。协议只定义数据和生命周期，不隐含线程模型。

## 输入与输出

```python
class FrameSource:
    def open(self, uri: str) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...

class ResultSink:
    def open(self, destination: str) -> None: ...
    def write(self, result: TrackResult) -> None: ...
    def finalize(self, expectedFrameCount: int) -> None: ...
```

`AirSim360DataSource` 使用 `open(root, sequenceId=None)`；读取器必须返回连续 `frameIndex`，到达末尾返回 `None`。

## 几何与后端

`SphericalGeometry` 提供 ERP/BFoV/局部视图之间的转换。控制器传给几何模块的所有搜索 `ViewSpec` 都是固定 `120° × 120°`，输出尺寸由 `geometry.viewWidthPx` 和 `geometry.viewHeightPx` 固定；禁止按轮次缩放视域。

`TrackerBackend` 提供：

```python
initialize(template: LocalView, templateBox: BBoxXYWH) -> None
infer(views: Sequence[LocalView], command: TemplateCommand)
    -> Sequence[LocalObservation]
close() -> None
```

每个 `ProjectedObservation` 必须带唯一 `viewId`、ERP/BFoV 框、局部框和 `fusedScore`。`StateEvaluator` 使用该 `fusedScore` 计算局部候选置信度，不用其他诊断分数替代。

## 控制器事务接口

控制器先执行 `buildInitialization()` 与 `commitInitialization()`，之后每帧使用 `beginFrame()`/`plan()` 获取 `SearchPlan`，把对应视域的观测交给 `consume()`。返回值为 `FrameCommitted` 或 `MoreViewsRequired`。

`SearchPlan.attemptIndex` 从 0 开始。调用方必须按计划中的 `viewId` 顺序返回观测；重复、未知或乱序响应会触发 `ProtocolError`。同一帧最终只提交一次 `StateObservation` 和一个 `TrackResult`。

## 评估数据契约

`StateObservation` 记录 `successRate`、`overlapThreshold`、`fusionSourceMinConfidence`、当前轮融合阈值、候选和 FuseBox 的来源视域、融合重合率、源框最低置信度、证据等级、是否最终轮和是否建议升级。FuseBox 只允许两个源框，且每个源局部框最多参加一个 FuseBox。

## 比赛协议

比赛专用实现位于 `src/instatarget/app/competition.py`：`OpenCvVideoSource` 解码 `.mp4`，`loadInitialBfov()` 解析初始 BFoV，`BfovResultSink` 按官方角度顺序原子写出每帧结果，`runCompetition()` 负责逐序列调度。
