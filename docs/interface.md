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

`AirSim360DataSource` 使用 `open(root, sequenceId=None)`，其余读取行为与 `FrameSource` 一致。读取器必须返回连续的 `frameIndex`，到达末尾返回 `None`。

## 几何与后端

`SphericalGeometry` 提供 ERP 框到 BFoV、BFoV 到局部视图、局部框到 BFoV 以及 BFoV 到 ERP 框的转换。`DepthProcessor` 只计算深度摘要和局部深度统计，不作目标决策。

`TrackerBackend` 提供：

```python
initialize(template: LocalView, templateBox: BBoxXYWH) -> None
infer(views: Sequence[LocalView], command: TemplateCommand)
    -> Sequence[LocalObservation]
close() -> None
```

`HiTSession` 是模型适配器的最小接口：`encodeTemplate()`、`infer()` 和 `close()`。`HiTBackend` 负责输入校验、异常翻译和输出类型校验；生产实现为 `PyTorchHiTSession`。

## 控制器

控制器先执行 `buildInitialization()` 与 `commitInitialization()`，之后每帧使用 `beginFrame()`/`plan()` 获取 `SearchPlan`，将后端观测交给 `consume()`。返回值为 `FrameCommitted` 或 `MoreViewsRequired`。后一种结果表示同一帧需要一次有界的额外视图搜索。

## 比赛协议

比赛专用实现位于 `src/instatarget/app/competition.py`，不依赖通用的像素框文本适配器：

- `OpenCvVideoSource`：持久化解码 `.mp4` 并输出 RGB `FramePacket`。
- `loadInitialBfov()`：解析 `init.txt` 的四个角度值。
- `BfovResultSink`：按官方 BFoV 角度顺序写出每帧一行并原子发布。
- `runCompetition()`：读取数据根目录、逐序列运行并输出进度。
