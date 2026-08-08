# InstaTargetingSystem 模块接口契约

> 本文档定义 AirSim360 输入、核心数据类型、模块 API、线程消息与错误语义。
> 示例使用 Python 类型标注表达契约；具体后端可用 PyTorch、ONNX Runtime 或 TensorRT。

---

## 1. 项目硬约束

| 项目 | 契约 |
|------|------|
| 帧顺序 | `frameIndex` 从 0 连续递增，禁止跳帧和重排 |
| 初始框 | 属于第 0 帧，格式为像素 `xywh` |
| 数据入口 | AirSim360 入口必须至少提供 ERP RGB；Depth 可缺省 |
| 内部位置 | 使用单位球面坐标、BFoV 和可选深度状态 |
| 跟踪模型 | 当前第三阶段 HiT 只接收局部透视 RGB 图；深度分支为第四阶段预留 |
| 深度使用 | 当前不参与后端主链路；未来深度预处理、第二个 HiT 和 MLP 融合统一封装进 `TrackerBackend` |
| 输出 | 每个输入帧恰好一个 `TrackResult` |
| 日志 | 比赛输出与日志分离；日志写 `stderr` |
| 配置 | 启动时完成校验，运行期间只读 |
| 外部格式 | 仅 `CompetitionAdapter` 可解释官方协议 |

---

## 2. AirSim360 数据约定

AirSim360 全景数据按同一帧对齐读取：

| 字段 | 内容 | 用途 |
|------|------|------|
| `frame_rgb_erp` | ERP 彩色全景图，RGB `uint8` | HiT 模板和搜索图来源 |
| `frame_depth` | 每像素距离值 | 目标距离、尺度和恢复门控 |
| `frame_semantic_mask` | 每像素类别 ID | 目标类别过滤和伪标注 |
| `frame_instance_mask` | 每像素实例 ID | 从 mask 生成临时 bbox |
| `semantic_class_list` | 类别 ID 到名称表 | 解释语义类别 |

同一像素在 RGB、Depth、semantic、instance 中表示同一球面方向。若某模态缺失，
必须显式标记，不允许填充伪造值冒充真实观测。

---

## 3. 坐标与数值约定

| 类型 | 约定 |
|------|------|
| ERP 像素 | 原点左上；`x` 向右，`y` 向下；连续浮点坐标 |
| 局部像素 | 原点左上；相对于透视视图 |
| 角度 | 弧度；`yaw ∈ [-π, π)`，`pitch ∈ [-π/2, π/2]` |
| 旋转 | 右手系；`+yaw` 向右，`+pitch` 向上 |
| 框 | `x, y, width, height`；宽高必须为正 |
| 图像 | RGB、`uint8`、连续 HWC；模型内部自行归一化 |
| 深度 | `float32`，单位由数据源声明；无效值用 mask 表示 |
| 置信度 | `[0, 1]`；越大越可信 |
| 时间 | 单调时钟纳秒，不使用墙钟计算性能 |

所有角度字段名必须带 `Rad`，时间字段必须带 `Ns` 或 `Ms`，像素字段必须带 `Px`，
深度字段必须带 `Depth` 或 `Range`。

---

## 4. 核心数据类型

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType

FrameIndex = NewType("FrameIndex", int)
SequenceId = NewType("SequenceId", str)


@dataclass(frozen=True, slots=True)
class BBoxXYWH:
    xPx: float
    yPx: float
    widthPx: float
    heightPx: float


@dataclass(frozen=True, slots=True)
class SphericalPoint:
    x: float
    y: float
    z: float
    yawRad: float
    pitchRad: float


@dataclass(frozen=True, slots=True)
class BFoV:
    center: SphericalPoint
    horizontalFovRad: float
    verticalFovRad: float
    rollRad: float = 0.0


@dataclass(frozen=True, slots=True)
class DepthPlane:
    values: "NDArray[float32]"      # shape [H, W]
    validMask: "NDArray[bool]"      # shape [H, W]
    unit: str                       # e.g. "m"


@dataclass(frozen=True, slots=True)
class SegmentationPlane:
    semantic: "NDArray[int32] | None"  # shape [H, W]
    instance: "NDArray[int32] | None"  # shape [H, W]
    classNames: dict[int, str]


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    timestampNs: int
    rgb: "NDArray[uint8]"                # shape [H, W, 3]
    depth: DepthPlane | None = None
    segmentation: SegmentationPlane | None = None


@dataclass(frozen=True, slots=True)
class DepthSummary:
    medianDepth: float
    meanDepth: float
    validRatio: float
    minDepth: float
    maxDepth: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MotionState3D:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    rangeDepth: float
    rangeVelocity: float
    confidence: float


class TrackStatus(Enum):
    TRACKING = auto()
    UNCERTAIN = auto()
    RECOVERING = auto()
    LOST = auto()


@dataclass(frozen=True, slots=True)
class TrackResult:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    bfov: BFoV
    confidence: float
    status: TrackStatus
    valid: bool
    depthSummary: DepthSummary | None = None
```

数据对象创建后不可变。大图像和深度图允许只读引用传递，其底层缓冲区生命周期必须覆盖消费者。

---

## 5. 框不变式与经线协议

### 5.1 内部不变式

- `0 <= xPx < frameWidthPx`，`0 <= yPx < frameHeightPx`。
- `widthPx > 0`，`heightPx > 0`。
- 普通框满足 `xPx + widthPx <= frameWidthPx`。
- 跨经线框允许 `xPx + widthPx > frameWidthPx`，表示水平循环区间。
- `heightPx <= frameHeightPx`；垂直方向不循环。
- BFoV 的水平/垂直 FOV 均在配置的开区间内。
- `SphericalPoint.xyz` 必须为单位向量，并与 yaw/pitch 一致。
- `DepthSummary.validRatio == 0` 时不得更新深度运动状态。

### 5.2 外部适配

官方格式若不接受循环框，由 `CompetitionAdapter` 采用官方约定执行以下一种策略：

1. 保留越界宽度表达；
2. 移动 x 到官方允许的展开坐标；
3. 选择左右两段中官方定义的主框；
4. 输出官方支持的双框结构。

核心模块不得猜测该策略。官方接口公布后，只修改适配器及其契约测试。

---

## 6. 全景几何接口

```python
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ViewSpec:
    viewId: int
    bfov: BFoV
    outputWidthPx: int
    outputHeightPx: int


@dataclass(frozen=True, slots=True)
class LocalView:
    spec: ViewSpec
    rgb: "NDArray[uint8]"
    depth: DepthPlane | None = None


class SphericalGeometry(Protocol):
    def bboxToBfov(
        self,
        bbox: BBoxXYWH,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> BFoV: ...

    def cropViews(
        self,
        frame: FramePacket,
        specs: Sequence[ViewSpec],
    ) -> Sequence[LocalView]: ...

    def localBoxToBfov(
        self,
        localBox: BBoxXYWH,
        spec: ViewSpec,
    ) -> BFoV: ...

    def bfovToBbox(
        self,
        bfov: BFoV,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> BBoxXYWH: ...
```

`cropViews()` 保持输入 `specs` 顺序。水平采样必须循环；空输入返回空列表；任何非有限
坐标抛出 `GeometryError`。若输入帧含深度，局部视图应携带同视角深度裁剪。

---

## 7. 深度与运动接口

```python
class DepthProcessor(Protocol):
    def summarize(
        self,
        frame: FramePacket,
        bbox: BBoxXYWH,
    ) -> DepthSummary | None: ...

    def summarizeLocal(
        self,
        view: LocalView,
        localBox: BBoxXYWH,
    ) -> DepthSummary | None: ...


class MotionEstimator(Protocol):
    def initialize(
        self,
        point: SphericalPoint,
        depth: DepthSummary | None,
        timestampNs: int,
    ) -> MotionState3D: ...

    def predict(
        self,
        timestampNs: int,
    ) -> MotionState3D: ...

    def update(
        self,
        point: SphericalPoint,
        depth: DepthSummary | None,
        timestampNs: int,
        observationConfidence: float,
    ) -> MotionState3D: ...
```

`DepthProcessor` 是第四阶段预留协议。当前第三阶段可以保留类型定义，但运行链路不调用它；
`DepthSummary` 只作为结果和消息字段中的可选占位。

`MotionEstimator` 可用常速度模型或 Kalman Filter。无深度时只更新球面方向；深度无效时
不得用默认距离污染速度。控制层内部按最近 `n` 帧窗口维护预测状态，单帧只是窗口中的一项观测，
不是独立决策依据。

---

## 8. Tracker 后端接口

```python
@dataclass(frozen=True, slots=True)
class LocalObservation:
    viewId: int
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    depthScore: float
    fusedScore: float
    depthSummary: DepthSummary | None
    latencyNs: int


class TemplateCommandKind(Enum):
    KEEP = auto()
    UPDATE_RECENT = auto()
    UPDATE_STABLE = auto()
    RESET_TO_ANCHOR = auto()


@dataclass(frozen=True, slots=True)
class TemplateCommand:
    kind: TemplateCommandKind
    frameIndex: FrameIndex
    viewId: int | None
    localBox: BBoxXYWH | None
    expectedRevision: int


class TrackerBackend(Protocol):
    def initialize(
        self,
        template: LocalView,
        templateBox: BBoxXYWH,
    ) -> None: ...

    def infer(
        self,
        views: Sequence[LocalView],
        command: TemplateCommand,
    ) -> Sequence[LocalObservation]: ...

    def close(self) -> None: ...
```

后端契约：

- `initialize()` 每个序列恰好调用一次；重复调用必须先 `close()`。
- `infer()` 输出与输入视图一一对应且顺序一致。
- 局部框必须已裁剪到视图有效区域。
- 当前第三阶段的实现只消费 `LocalView.rgb`，并将 `depthScore` 固定为 `0.0`、`fusedScore` 固定为 `appearanceScore`、`depthSummary` 固定为 `None`。
- 第四阶段若启用 RGB-D，后端内部可以读取深度并完成深度预处理、编码与融合，但不得生成 BFoV、改变状态机或执行全局搜索规划。
- 同一后端实例只允许设备线程调用。
- 不支持在线模板的后端必须只接受 `KEEP`，其能力在启动时声明。

---

## 9. 控制器接口

```python
@dataclass(frozen=True, slots=True)
class SearchPlan:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    views: tuple[ViewSpec, ...]
    templateCommand: TemplateCommand
    predictedMotion: MotionState3D | None


@dataclass(frozen=True, slots=True)
class InitializationPlan:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    templateView: ViewSpec
    templateBox: BBoxXYWH


@dataclass(frozen=True, slots=True)
class ProjectedObservation:
    viewId: int
    bfov: BFoV
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    motionScore: float
    scaleScore: float
    depthScore: float
    fusedScore: float
    depthSummary: DepthSummary | None


class TrackController(Protocol):
    def buildInitialization(
        self,
        frame: FramePacket,
        initialBox: BBoxXYWH,
    ) -> InitializationPlan: ...

    def commitInitialization(
        self,
        plan: InitializationPlan,
        depthSummary: DepthSummary | None,
    ) -> TrackResult: ...

    def plan(self, frame: FramePacket) -> SearchPlan: ...

    def update(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> TrackResult: ...
```

`buildInitialization()` 生成首帧模板视图和该视图中的模板框；
`commitInitialization()` 仅在后端初始化成功后提交第 0 帧。`plan()` 必须使用最近 `n` 帧的
运动状态、深度摘要和置信度生成搜索视图。`update()` 必须校验帧号和 revision，并原子提交
状态；失败时原状态保持不变。控制层在 `update()` 内只做候选选择、状态更新和轻量门控。
当前第三阶段没有深度神经网络或 MLP 融合；第四阶段启用后也不得把这些后端计算上移到控制层。

`ProjectedObservation.depthScore` 和 `ProjectedObservation.fusedScore` 由 `TrackerBackend` 产生；
`motionScore` 与 `scaleScore` 由控制层补充。

---

## 10. AirSim360 数据接口

```python
@dataclass(frozen=True, slots=True)
class AirSim360Record:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    rgbPath: str
    depthPath: str | None
    semanticPath: str | None
    instancePath: str | None


class AirSim360DataSource(Protocol):
    def open(self, root: str, sequenceId: str) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


class PseudoTrackBuilder(Protocol):
    def buildInitialBox(
        self,
        frame: FramePacket,
        targetInstanceId: int,
    ) -> BBoxXYWH: ...

    def buildPseudoGroundTruth(
        self,
        frame: FramePacket,
        targetInstanceId: int,
    ) -> tuple[BBoxXYWH, bool]: ...
```

`PseudoTrackBuilder` 只用于训练、验证和回归测试。正式比赛推理仍以官方初始框为准。
若 AirSim360 的实例 ID 不跨帧稳定，必须通过外观和位置连续性生成临时 `trackId`。

---

## 11. 线程消息接口

```python
@dataclass(frozen=True, slots=True)
class InitRequest:
    plan: InitializationPlan
    frame: FramePacket


@dataclass(frozen=True, slots=True)
class InitResponse:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    depthSummary: DepthSummary | None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    plan: SearchPlan
    frame: FramePacket


@dataclass(frozen=True, slots=True)
class InferResponse:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    observations: tuple[LocalObservation, ...]
    depthSummaries: dict[int, DepthSummary]


@dataclass(frozen=True, slots=True)
class ResultPacket:
    result: TrackResult
    totalLatencyNs: int


@dataclass(frozen=True, slots=True)
class FatalError:
    stage: str
    message: str
    frameIndex: FrameIndex | None
```

当前第三阶段的 `InferResponse.depthSummaries` 必须为空字典；深度摘要字典只给第四阶段预留。

所有请求和响应必须携带帧号与 revision。`T0` 不接受旧响应；工作线程不捕获并吞掉
异常，而是转换为一次 `FatalError`。

---

## 12. 输入输出接口

```python
class FrameSource(Protocol):
    def open(self, uri: str) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


class ResultSink(Protocol):
    def open(self, destination: str) -> None: ...
    def write(self, result: TrackResult) -> None: ...
    def finalize(self, expectedFrameCount: int) -> None: ...
```

- `read()` 在正常 EOF 返回 `None`；解码失败抛出 `DecodeError`。
- `write()` 只接受严格递增帧号。
- `finalize()` 校验结果数量后原子发布文件。
- 中间文件使用 `.partial` 后缀，异常时不得冒充最终结果。

开发期默认文本格式为每行：

```text
xPx,yPx,widthPx,heightPx
```

浮点数使用小数点，禁止科学计数法；精度固定为 6 位。正式比赛格式由
`CompetitionAdapter` 覆盖。

---

## 13. 应用入口

开发入口必须支持：

```bash
python -m instatarget.track \
  --input input.mp4 \
  --init-box 120.0,80.0,64.0,96.0 \
  --output result.txt \
  --config configs/RGBonly.yaml
```

AirSim360 离线入口必须支持：

```bash
python -m instatarget.track_airsim360 \
  --dataset-root data/AirSim360 \
  --sequence NYC_001 \
  --target-instance 305 \
  --output result.txt \
  --config configs/RGBonly.yaml
```

退出码：

| 退出码 | 含义 |
|--------|------|
| `0` | 成功，结果已完整发布 |
| `2` | 参数或配置错误 |
| `3` | 输入或解码错误 |
| `4` | 模型加载或推理错误 |
| `5` | 输出错误 |
| `10` | 内部不变式错误 |

比赛 Docker 入口由 `CompetitionAdapter` 将官方调用转换为同一内部 Driver API。

---

## 14. 配置契约

```yaml
schemaVersion: 1
model:
  backend: pytorch
  variant: hit_small
  weights: ../models/hit_small.pth
  precision: fp32
geometry:
  viewWidthPx: 256
  viewHeightPx: 256
  minFovDeg: 20.0
  maxFovDeg: 120.0
depth:
  enabled: false
  minValidRatio: 0.35
  maxDepthJumpRatio: 0.60
backendFusion:
  depthScoreWeight: 0.0
decisionGate:
  motionScoreWeight: 0.25
  scaleScoreWeight: 0.15
tracking:
  acceptThreshold: 0.70
  uncertainThreshold: 0.45
  stableFramesBeforeUpdate: 8
  windowLength: 5
recovery:
  maxViewsPerFrame: 12
  globalSearchInterval: 5
runtime:
  decodeQueueCapacity: 3
  inferRequestQueueCapacity: 1
  inferResponseQueueCapacity: 1
  resultQueueCapacity: 32
```

未知字段默认报错；相对路径以配置文件目录为基准；角度配置可用 `Deg`，加载后立即转换为
`Rad`。阈值必须满足 `0 <= uncertainThreshold < acceptThreshold <= 1`；`windowLength >= 2`；
所有队列容量必须为正整数。

---

## 15. 错误与诊断

```python
class InstaTargetError(Exception): ...
class ConfigError(InstaTargetError): ...
class DecodeError(InstaTargetError): ...
class GeometryError(InstaTargetError): ...
class DepthError(InstaTargetError): ...
class ModelError(InstaTargetError): ...
class ProtocolError(InstaTargetError): ...
class OutputError(InstaTargetError): ...
```

- 用户输入错误使用明确异常，不使用 `assert`。
- 内部不变式可使用 `assert`，生产入口必须转换为退出码 `10`。
- 日志至少含序列、帧号、状态和阶段，不含整帧像素、深度矩阵或模型权重内容。
- 比赛模式的标准输出只允许官方结果；日志统一写标准错误。

---

## 16. 后端一致性契约

同一固定输入上，部署后端必须与 PyTorch FP32 参考结果比较：

- 输出数量、顺序和 `viewId` 完全一致。
- 框坐标最大绝对误差阈值由配置声明。
- 分数误差不得引起状态阈值两侧翻转；临界样本使用 FP32 或安全边距。
- 导出模型的权重哈希、输入尺寸和预处理版本写入运行清单。

任何一致性测试失败都阻止生成提交镜像。

---

## 17. 版本兼容

- 配置、线程消息和外部结果格式分别维护 `schemaVersion`。
- 同一主版本内只允许增加可选字段。
- 删除字段、修改单位或改变经线语义必须提升主版本。
- 反序列化器拒绝未知主版本，不做猜测性兼容。
