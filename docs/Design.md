# InstaTargetingSystem 总体设计

> 系统以 **360VOT** 处理球面几何，以 **TrackerBackend** 完成深度图像处理 + HiT + MLP，
> 以 **DepthAwareTrackController**（简称 **DTC**）统一多帧预测、深度门控和结果选择。

---

## 1. 目标

| 项目 | 目标 |
|------|------|
| 输入 | AirSim360 全景序列或等价 ERP 视频 + 初始框 |
| 输出 | 每帧目标框、状态、置信度；比赛模式只输出官方格式 |
| 主线 | `360VOT + TrackerBackend + DTC` |
| 深度 | 作为后端内部处理链路的一部分，与 HiT 和 MLP 一起封装 |
| 退化 | 同时支持 `rgb_depth` 与 `rgb_only` |
| 训练 | 只训练后端中的深度模块和融合头，冻结主 RGB 主干 |
| 实时性 | 常规帧单视图，低置信时才扩窗或全景搜索 |

系统不把整张 ERP 直接送入后端主干。后端只看低畸变局部视图；球面状态由 DTC 统一维护。

---

## 2. 选型

| 层级 | 选型 | 作用 |
|------|------|------|
| 几何 | 360VOT 风格 BFoV | ERP / 局部视图 / 回投影 |
| 跟踪后端 | TrackerBackend | 深度预处理 + HiT + MLP |
| RGB 主干 | HiT-Small | 主 RGB 跟踪 |
| 加速备选 | HiT-Tiny / DyHiT | 更轻或更快的后端变体 |
| 控制层 | DTC | 多帧预测、深度门控、恢复规划 |
| 深度初始化 | Depth-Anything-V2-Small | 深度模块 warm start |
| 融合头 | 后端内 MLP | 融合 RGB、深度、运动和尺度 |

LightTrack 只作为轻量 baseline。360VOT 是全景表示与评测基础，不是主跟踪器。

---

## 3. 主架构

```text
FrameSource
   -> geometry(BFoV / sync crop)
   -> DTC(window state + depth gate)
   -> TrackerBackend(depth + HiT + MLP)
   -> ResultSink
```

| 模块 | 职责 |
|------|------|
| `geometry` | 同步裁剪 RGB 和 Depth，生成局部视图 |
| `DTC` | 维护状态、做多帧预测、生成搜索计划 |
| `TrackerBackend` | 深度处理、局部 RGB 跟踪、MLP 融合 |
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
- 最近 `n` 帧深度摘要；
- 最近 `n` 帧 BFoV 深度块；
- 最近 `n` 帧置信度；
- 最近 `n` 帧球面方向变化率。

处理顺序：

1. 将方向转成单位向量序列。
2. 将深度转成相对距离序列。
3. 用常速度 / Alpha-Beta / Kalman 预测下一帧中心。
4. 结合深度跳变决定下一帧搜索中心和 FOV。
5. 输出 `SearchPlan` 和 `TemplateCommand`。

### 4.3 后端内部融合

`TrackerBackend` 内部完成：

1. 深度图预处理。
2. RGB 局部图编码。
3. 深度特征编码。
4. HiT 局部匹配。
5. MLP 融合输出 `fusedScore`。

融合优先级：

1. 特征级融合；
2. 分数级融合；
3. 双流完整 HiT 不采用。

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

1. 用 `Depth-Anything-V2-Small` 初始化后端深度模块。
2. 冻结主 RGB 主干。
3. 只训练后端深度模块和融合 MLP。
4. 优先在 `rgb_depth` 上训练，再用 `rgb_only` 验证退化路径。

### 5.3 训练阶段

| 阶段 | 内容 | 产物 |
|------|------|------|
| A | 跑通 HiT 基线 | 可复现普通视频结果 |
| B | 接入 BFoV | 全景几何基线 |
| C | 接入深度摘要 | 深度门控基线 |
| D | 训练深度模块 + MLP | 主线融合权重 |
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
- `DTC` 使用最近 `n` 帧预测下一帧。
- `HiT` 主干只走 RGB。
- 后端深度模块和融合头可独立训练。
- `rgb_depth` 与 `rgb_only` 都能跑通。
