# InstaTargetingSystem 总体设计

> 当前基线以 **360VOT** 处理球面几何，以 **TrackerBackend** 完成 RGB-only/RGB-D HiT 观测、
> 深度处理、融合和模板协议，以 **DepthAwareTrackController**（简称 **DTC**）统一多视图计划、
> 多帧预测、候选聚合、结果选择和恢复。控制层只消费后端输出，不把融合逻辑上移。

---

## 1. 目标

| 项目 | 目标 |
|------|------|
| 输入 | AirSim360 全景序列或等价 ERP 视频 + 初始框 |
| 输出 | 每帧目标框、状态、置信度；比赛模式只输出官方格式 |
| 主线 | `360VOT + geometry + TrackerBackend(RGB/RGB-D) + DTC` |
| 深度 | geometry 同步裁剪；TrackerBackend 内完成深度伪彩色、深度分支和融合头 |
| 退化 | `depth.enabled=false` 时严格退化为 `rgb_only`，输出契约不变 |
| 训练 | HiT 主干可冻结；深度分支和融合头独立训练并回灌后端 |
| 实时性 | 状态相关五视图/环搜/cube-map；低分时最多一次同帧有界升级 |

系统不把整张 ERP 直接送入后端主干。后端只看低畸变局部视图；球面状态由 DTC 统一维护。

---

## 2. 选型

| 层级 | 选型 | 作用 |
|------|------|------|
| 几何 | 360VOT 风格 BFoV | ERP / 局部视图 / 回投影 |
| 跟踪后端 | TrackerBackend | RGB HiT + 可选深度分支 + 融合头；只输出局部观测 |
| RGB 主干 | HiT-Small | 主 RGB 跟踪 |
| 加速备选 | HiT-Tiny / DyHiT | 更轻或更快的后端变体 |
| 控制层 | DTC | 负责多视图计划、单帧候选聚合、多帧预测、恢复和模板策略 |
| 深度初始化 | Depth-Anything-V2-Small | 深度编码器 warm start 候选 |
| 融合头 | 后端内 MLP | 融合 RGB、深度、模板上下文和轻量几何参数 |

LightTrack 只作为轻量 baseline。360VOT 是全景表示与评测基础，不是主跟踪器。

---

## 3. 主架构

```text
FrameSource
   -> geometry(BFoV / sync crop)
   -> DTC(window state + multi-view plan)
   -> TrackerBackend(RGB/RGB-D HiT + fusion)
   -> ResultSink
```

人工诊断时，应用编排层可选择把各阶段的只读结果旁路到 `visualization`：

```text
geometry local RGB ----------------------\
existing depth-to-RGB result -------------+-> visualization -> lossless PNG
TrackerBackend local observations --------+
geometry projected observations ----------/
```

该旁路默认关闭，不进入跟踪闭环，也不改变任何计算结果。深度 RGB 由 `TrackerBackend` 的
深度链路生成，`visualization` 只读取并原样保存，不维护第二套深度颜色化实现。

后端拓扑是：

```text
local RGB -> HiT-RGB ----\
                          -> MLP fusion -> fusedScore
depth -> relief color -> HiT-Depth -----/
```

| 模块 | 职责 |
|------|------|
| `geometry` | 同步裁剪 RGB 和 Depth，生成局部视图 |
| `DTC` | 维护状态、做多帧预测、生成搜索计划 |
| `TrackerBackend` | 局部 RGB/RGB-D 跟踪、深度处理、后端融合与模板执行 |
| `DTC` | 候选排序、单帧聚合、状态门控与全局恢复 |
| `ResultSink` | 输出比赛格式和诊断日志 |
| `visualization` | 可选保存局部 RGB、已有深度 RGB、后端局部框和 geometry ERP 框 |

模块依赖方向固定为：

```text
core <- geometry <- tracker <- controller <- app
                     ^              |
                     +--------------+

core <- visualization <- app
```

`geometry`、`tracker` 和 `controller` 不依赖 `visualization`。所有采集点由 `app` 调用，保证关闭
可视化后现有模块框架和数据流不变。输出统一为无损 PNG，框颜色固定为荧光绿 `#39FF14`；目录、
记录项选择和使用方法见 `docs/visualization.md`。

---

## 4. 核心算法

### 4.1 360VOT 几何

- 以 BFoV 作为统一中介。
- 首帧框转 BFoV，BFoV 再转局部透视图。
- 局部框回投影时采样边界，不只采四角。
- RGB 与 Depth 必须同视角裁剪。

### 4.2 DTC 多帧预测

`DTC` 不靠单帧决策，而是使用最近 `n` 个可靠测量窗口，默认 `n=5`。纯预测输出不进入窗口。

输入窗口包含：

- 最近 `n` 帧球面中心；
- 最近 `n` 帧置信度；
- 最近 `n` 帧球面方向变化率。

处理顺序：

1. 将方向转成单位向量序列。
2. 在最后可靠中心的球面切平面做稳健常速度拟合，样本不足时退化为零速度/Alpha-Beta。
3. 独立拟合对数尺度和可选 log-range，并估计随缺失增长的不确定度。
4. 结合连续性、尺度变化、历史置信度和不确定度决定搜索中心和 FOV。
5. 输出带 transaction/attempt 的 `SearchPlan` 和 `TemplateCommand`。

深度摘要由 TrackerBackend 生成后进入控制闭环；DTC 不再处理整张深度图。

### 4.3 后端内部融合

`TrackerBackend` 内部固定执行：

1. 深度图先做对齐、归一化、背景浮雕化和轮廓增强，再转成单调伪彩色。
2. 同一局部视图再走一条深度 HiT 分支。
3. 将 RGB HiT 输出、深度 HiT 输出、模板上下文和轻量几何参数送入 MLP。
4. 融合头单独训练，初值偏向 RGB，深度作为辅助判别分支；`fusedScore` 由后端统一产生。

### 4.4 状态机与恢复

```text
INIT -> TRACKING -> UNCERTAIN -> RECOVERING -> TRACKING
                       |             |
                       +-----------> LOST
                                      |
                                      +-> RECOVERING
```

| 状态 | 行为 | 模板更新 |
|------|------|----------|
| `TRACKING` | 主视图 + 四个角保护视图，稳健候选聚合 | 允许稳定更新 |
| `UNCERTAIN` | 五视图扩窗，可同帧升级一次 | 禁止 |
| `RECOVERING` | 搜索种子 + 跨帧去重环搜 | 禁止 |
| `LOST` | 降频六面 cube-map 并输出预测 | 禁止 |

`StateEvaluator` 只融合一个球面一致候选簇，不把不相交候选做并集。恢复阶段只改变搜索范围和
预测假设，不改变职责边界；`maxAttemptsPerFrame` 从结构上保证同一帧不会无限重试。

---

## 5. 数据与训练

### 5.1 数据源

| 数据源 | 用途 |
|------|------|
| AirSim360 | 主训练与回归测试 |
| 真实全景数据 | 泛化验证 |
| 360VOT | 全景几何与评测 |
| 常规 SOT 数据 | 只补 HiT 基础能力 |

### 5.2 训练策略

1. RGB-only 和 RGB-D 共用同一套 `LocalView`/`LocalObservation` 契约。
2. 深度分支优先复用伪彩色深度编码初始化，而不是从零学习完整深度表征。
3. 融合头单独训练，先固定两个 HiT 分支，再微调融合参数。
4. 先在 `rgb_depth` 上校准，再回到 `rgb_only` 验证退化路径不崩。

### 5.3 训练阶段

| 里程碑 | 内容 | 产物 |
|------|------|------|
| A | 跑通 HiT 基线 | 可复现普通视频结果 |
| B | 接入 BFoV | 全景几何基线 |
| C | RGB-only 后端与模板协议 | 已完成基线 |
| D | 深度伪彩色 + 双 HiT + MLP | 已完成 RGB-D 后端 |
| E | 多视图 DTC、候选聚合、恢复和模板门控 | DTC 控制层 |
| F | 导出 ONNX / TensorRT | 部署权重 |

---

## 6. 评测

### 6.1 主指标

- `AUC`
- `Success Rate@0.5`
- `FPS`

### 6.2 内部指标

- 球面角度精度
- 找回成功率
- 误找回率
- `TRACKING / RECOVERING / LOST` 占比
- 深度一致性命中率

---

## 7. 工程结构

```text
InstaTargetingSystem/
  configs/
  docker/
  docs/
  models/
  src/instatarget/
    app/
    core/
    geometry/
    tracker/
    controller/
    io/
    eval/
    visualization/
```

---

## 8. 实施顺序

1. 先跑通 `TrackerBackend + 360VOT` 基线。
2. 再接入 `geometry` 的同步裁剪。
3. 已完成 RGB-D 后端、深度摘要和融合头，并验证 RGB-only 退化。
4. 已接入 V2 DTC 的帧事务、状态评估器、窗口预测、状态相关视图和恢复记忆。
5. 最后做 app/io、评测、训练回灌、部署和性能优化。

---

## 9. 完成定义

- `geometry` 同步输出 RGB 和 Depth。
- `DTC` 只消费 TrackerBackend 的局部观测、融合分数和深度摘要，不实现后端融合。
- `HiT` RGB 主干保留；深度分支通过后端可选接入。
- 深度颜色化与融合头已实现，并可在独立训练链路中更新权重。
- `rgb_only` 和 `rgb_depth` 均保持同一接口可运行。
- DTC 完成定义还包括每帧唯一提交、稳健单簇聚合、最多一次同帧升级和有界恢复。
- 可按配置选择记录四类中间图，且关闭可视化时不改变原计算链路。
