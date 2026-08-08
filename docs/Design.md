# InstaTargetingSystem 总体设计

> 当前第三阶段以 **360VOT** 处理球面几何，以 **RGB-only TrackerBackend** 完成 HiT 观测和模板协议，
> 以 **DepthAwareTrackController**（简称 **DTC**）统一多帧预测、结果选择和后续深度预留接口。
> 深度伪彩色 + 双 HiT + MLP 融合属于第四阶段预留方案，尚未进入当前代码实现。

---

## 1. 目标

| 项目 | 目标 |
|------|------|
| 输入 | AirSim360 全景序列或等价 ERP 视频 + 初始框 |
| 输出 | 每帧目标框、状态、置信度；比赛模式只输出官方格式 |
| 主线 | `360VOT + RGB-only TrackerBackend + DTC` |
| 深度 | 当前实现不进入后端主链路，第四阶段预留深度伪彩色双流方案 |
| 退化 | 当前阶段只跑 `rgb_only`；`rgb_depth` 作为后续扩展目标 |
| 训练 | 当前只保持 HiT 既有能力；深度分支和融合头单独设计、单独训练 |
| 实时性 | 常规帧单视图，低置信时才扩窗或全景搜索 |

系统不把整张 ERP 直接送入后端主干。后端只看低畸变局部视图；球面状态由 DTC 统一维护。

---

## 2. 选型

| 层级 | 选型 | 作用 |
|------|------|------|
| 几何 | 360VOT 风格 BFoV | ERP / 局部视图 / 回投影 |
| 跟踪后端 | TrackerBackend | 当前为 RGB-only HiT；深度伪彩色 + 双 HiT + MLP 预留 |
| RGB 主干 | HiT-Small | 主 RGB 跟踪 |
| 加速备选 | HiT-Tiny / DyHiT | 更轻或更快的后端变体 |
| 控制层 | DTC | 当前负责多帧预测、候选选择和恢复规划；深度门控预留 |
| 深度初始化 | Depth-Anything-V2-Small | 第四阶段深度模块 warm start 候选 |
| 融合头 | 后端内 MLP | 第四阶段预留；融合 RGB、深度、模板上下文和轻量几何参数 |

LightTrack 只作为轻量 baseline。360VOT 是全景表示与评测基础，不是主跟踪器。

---

## 3. 主架构

```text
FrameSource
   -> geometry(BFoV / sync crop)
   -> DTC(window state + RGB gate)
   -> TrackerBackend(RGB-only HiT)
   -> ResultSink
```

第四阶段预留的后端拓扑是：

```text
local RGB -> HiT-RGB ----\
                          -> MLP fusion -> fusedScore
depth -> relief color -> HiT-Depth -----/
```

| 模块 | 职责 |
|------|------|
| `geometry` | 同步裁剪 RGB 和 Depth，生成局部视图 |
| `DTC` | 维护状态、做多帧预测、生成搜索计划 |
| `TrackerBackend` | 当前只负责局部 RGB 跟踪与模板执行；深度处理和 MLP 融合预留到下一阶段 |
| `fusion` | 候选排序与最终门控 |
| `ResultSink` | 输出比赛格式和诊断日志 |

模块依赖方向固定为：

```text
core <- geometry <- tracker <- controller <- app
                     ^              |
                     +--------------+
```

---

## 4. 核心算法

### 4.1 360VOT 几何

- 以 BFoV 作为统一中介。
- 首帧框转 BFoV，BFoV 再转局部透视图。
- 局部框回投影时采样边界，不只采四角。
- RGB 与 Depth 必须同视角裁剪。

### 4.2 DTC 多帧预测

`DTC` 不靠单帧决策，而是使用前 `n` 帧窗口，默认 `n=3~5`。

输入窗口包含：

- 最近 `n` 帧球面中心；
- 最近 `n` 帧置信度；
- 最近 `n` 帧球面方向变化率。

处理顺序：

1. 将方向转成单位向量序列。
2. 用常速度 / Alpha-Beta / Kalman 预测下一帧中心。
3. 结合连续性、尺度变化和历史置信度决定下一帧搜索中心和 FOV。
4. 输出 `SearchPlan` 和 `TemplateCommand`。

第四阶段才会把深度摘要、深度边缘和伪彩色深度分支并入控制闭环。

### 4.3 后端内部融合

当前第三阶段的 `TrackerBackend` 内部只完成：

1. RGB 局部图编码。
2. HiT 局部匹配。
3. 局部框裁剪与输出规范化。

第四阶段预留的后端融合思路是：

1. 深度图先做对齐、归一化、背景浮雕化和轮廓增强，再转成单调伪彩色。
2. 同一局部视图再走一条深度 HiT 分支。
3. 将 RGB HiT 输出、深度 HiT 输出、模板上下文和轻量几何参数送入 MLP。
4. 融合头单独训练，初值偏向 RGB，深度作为辅助判别分支。

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
| `TRACKING` | 单视图跟踪 | 允许稳定更新 |
| `UNCERTAIN` | 扩窗验证 | 禁止 |
| `RECOVERING` | 环搜 / 全景粗搜 | 禁止 |
| `LOST` | 降频搜索并输出预测 | 禁止 |

恢复阶段只改变搜索范围，不改变模块职责边界。

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

1. 当前第三阶段只验证 RGB-only 后端与模板协议。
2. 第四阶段如果启用深度分支，优先复用伪彩色深度编码初始化，而不是从零学完整深度表征。
3. 融合头单独训练，先固定两个 HiT 分支，再微调融合参数。
4. 先在 `rgb_depth` 上校准，再回到 `rgb_only` 验证退化路径不崩。

### 5.3 训练阶段

| 阶段 | 内容 | 产物 |
|------|------|------|
| A | 跑通 HiT 基线 | 可复现普通视频结果 |
| B | 接入 BFoV | 全景几何基线 |
| C | RGB-only 后端与模板协议 | 当前可运行版本 |
| D | 深度伪彩色 + 双 HiT + MLP | 第四阶段预留融合版本 |
| E | 加入恢复和模板门控 | 长时版本 |
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
```

---

## 8. 实施顺序

1. 先跑通 `TrackerBackend + 360VOT` 基线。
2. 再接入 `geometry` 的同步裁剪。
3. 再接入 `DTC` 的多帧预测。
4. 再训练后端深度模块和融合头。
5. 最后做恢复、部署和性能优化。

---

## 9. 完成定义

- `geometry` 同步输出 RGB 和 Depth。
- `DTC` 当前第三阶段只消费 RGB-only 后端输出，深度分支为预留设计。
- `HiT` 主干当前只走 RGB。
- 第四阶段的深度颜色化与融合头可以独立训练。
- `rgb_only` 当前已跑通；`rgb_depth` 仍是后续目标。
