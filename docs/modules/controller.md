# InstaTargetingSystem 控制器规范

> 本文定义 `DepthAwareTrackController`（DTC）的实现契约。`geometry`、`tracker` 和
> `visualization` 已完成的功能不在 DTC 中重做：geometry 负责同步投影/裁剪，TrackerBackend
> 负责 RGB-D 后端推理与融合，visualization 只做旁路记录。DTC 是 T0 线程中的唯一有状态模块，
> 负责多视图计划、单帧候选聚合、多帧运动预测、状态机、恢复和模板命令。

## 1. 职责边界

| 输入 | 输出 | 不负责 |
|---|---|---|
| `FramePacket`、最近窗口状态、`LocalObservation`/`ProjectedObservation`、可选 `DepthSummary` | `InitializationPlan`、`SearchPlan`、`TrackResult`、`TemplateCommand` | 图像裁剪、深度伪彩色、HiT 推理、RGB/Depth 融合、文件 I/O |

固定边界：

1. DTC 只消费后端输出的 `fusedScore`、`depthScore`、`DepthSummary` 和局部框，不读取整张深度图。
2. `fusedScore` 是 TrackerBackend 的模型融合结果；DTC 可计算轻量的 `decisionScore` 做排序和
   状态门控，但不能重做 MLP 或覆盖 `fusedScore`。
3. DTC 生成一个包含多个 `ViewSpec` 的计划，geometry 按顺序裁剪 RGB/Depth，TrackerBackend
   按相同顺序返回 `LocalObservation`。
4. 所有请求/响应校验 `sequenceId`、`frameIndex`、`stateRevision`。旧 revision、重复 viewId、
   乱序帧一律拒绝，原状态保持不变。

## 2. 结果术语

- **单图预测框**：一个局部视图对应的 `LocalObservation.bbox`。
- **单帧预测框**：当前帧所有候选过滤、聚类和加权融合后的 `FrameAggregate.bbox`（可先为
  controller 私有类型）。
- **多帧预测框**：结合历史窗口和运动模型得到的 `TrackResult.bbox`。没有可靠观测时，
  `TrackResult.valid=false`；有可靠候选确认时，`valid=true`。

单图分数不能直接当作单帧最终置信度。DTC 必须保留每个候选的 `viewId`、来源角色和各分数，
以便诊断和复现聚合过程。

## 3. 状态机

```text
INIT -> TRACKING -> UNCERTAIN -> RECOVERING -> TRACKING
                       |             |
                       +-----------> LOST
                                      |
                                      +-> RECOVERING
```

| 状态 | 进入条件 | 当前帧动作 | 模板 |
|---|---|---|---|
| `INIT` | 第 0 帧模板视图和初始框校验成功 | 建立 anchor、运动状态和目标尺度 | 只初始化 |
| `TRACKING` | `decisionScore >= acceptThreshold` 且候选通过运动/尺度门控 | guard triplet + 主预测视图 + 必要尺度视图 | 允许稳定更新 |
| `UNCERTAIN` | 低于接收阈值但不低于不确定阈值，或多视图不一致 | 放大上下文并保留有限预测假设 | 禁止更新 |
| `RECOVERING` | 连续 `uncertainPatience` 帧未确认 | 环搜/多假设搜索，按帧推进，不重复同一视图 | 禁止更新 |
| `LOST` | 超过 `maxRecoveryFrames` 或恢复预算耗尽 | 按间隔做全景粗搜，其余帧输出预测 | 禁止更新 |

状态转换使用滞回和连续帧计数，避免阈值附近抖动。找回时要求候选达到
`recoverAcceptThreshold`、至少满足 `minViewsForCommit` 个相互支持视图（初始化帧例外），
并通过运动/尺度门控。

## 4. 初始化协议

1. 读取第 0 帧和外部初始框，校验框尺寸及 ERP 边界。
2. 通过 geometry 将初始框转换为 BFoV，生成模板 `ViewSpec`；模板上下文的宽高至少为初始框的
   2 倍，受 geometry FOV 上下限约束。
3. 调用 `TrackerBackend.initialize(templateView, templateBox)`。RGB-D 模式下，后端同时缓存
   对齐的深度模板；无深度时保持 RGB-only。
4. DTC 初始化 `MotionEstimator` 和 anchor 模板，提交第 0 帧 `TrackResult`，状态转为 `TRACKING`。
5. 首帧结果必须是规范化后的初始框，不运行预测或候选竞争。

## 5. 每帧搜索计划

### 5.1 固定 guard triplet

每帧都生成三张保护视图，中心基于上一帧已提交框或主预测中心，yaw 偏移为 `-120°、0°、+120°`，
pitch 取预测 pitch。FOV 取不小于目标上下文所需值并限制在 geometry 的 `minFov/maxFov`。三图
允许重叠，目的是覆盖左右相邻球面区域、防止局部过拟合和经线附近丢失。

### 5.2 状态相关视图

- `TRACKING`：预测中心主视图；目标尺度变化明显时增加放大/缩小视图。
- `UNCERTAIN`：扩大上下文，并加入最多 `maxPredictionHorizon` 个角速度或深度变化假设。
- `RECOVERING`：按未覆盖的环搜索半径和方位生成视图，视图不得重复。
- `LOST`：每 `globalSearchInterval` 帧执行一次全景粗搜；非搜索帧只推进预测，不阻塞等待。

总视图数不得超过 `recovery.maxViewsPerFrame`，且该值必须 `>=3`。预算不足时优先级固定为：
`guard triplet > 主预测视图 > 预测假设 > 环搜/全景视图`。

### 5.3 上下文 FOV

对主预测框和每个恢复假设分别计算上下文：

```text
contextWidth  >= contextScale * max(initialWidth, predictedWidth)
contextHeight >= contextScale * max(initialHeight, predictedHeight)
```

`contextScale` 默认 `2.0`，再叠加 `contextMarginRatio`；这保证局部面积至少约为首框面积 4 倍。
目标运动导致预测框变大时，下一帧上下文必须同步扩大；达到 `maxFov` 时记录裁剪状态，不静默
缩小目标上下文。

## 6. 单帧候选聚合

### 6.1 回投影和过滤

应用层将 `LocalObservation` 的局部框通过 `geometry.localBoxToBfov()` 和现有 seam 规则回投影，
构造 `ProjectedObservation`，并补充：

- `motionScore`：候选 BFoV 中心与预测中心的球面连续性；
- `scaleScore`：候选尺寸与初始/历史尺寸的合理性；
- `depthConsistencyScore`：由 `DepthSummary` 的有效率、深度跳变和历史范围计算。

建议使用以下轻量选择分数，模态缺失时对有效权重重新归一化：

```text
decisionScore = normalize(
    w_backend * fusedScore
  + w_motion  * motionScore
  + w_scale   * scaleScore
  + w_depth   * depthConsistencyScore
)
```

`candidateMinScore` 默认建议为 `0.40`。低于该值的单图候选不参与聚类和最终框计算，但保留在
诊断日志中；DTC 不把低分硬改为零后继续平均。

### 6.2 聚类和最终框

1. 将候选中心转为单位向量，按球面角距离聚类；同时要求对数尺度差不超过
   `scaleClusterTolerance`。
2. 候选比较可以使用球面 IoU，或“中心角 + 尺度差”的确定性兼容实现；禁止直接用普通平面 IoU
   处理跨经线框。
3. 每簇以 `decisionScore` 为权重，采用加权中位数/截尾均值融合中心、宽高和深度，抑制离群框。
4. 选簇时综合总权重、最高分、视图覆盖度和运动连续性；单张图的高分不能自动压过多视图一致
   的中分簇。
5. 输出单帧预测框和单帧置信度，再交给状态机决定是否提交为有效的多帧结果。

首个可运行版本使用确定性聚合，不引入跨视图端到端网络。聚合权重在离线序列上校准，随后
固定到配置或模型版本中。

## 7. 多帧运动预测

`motion_estimator.py` 首先实现常速度 Alpha-Beta 模型；Kalman 作为可替换实现，不应阻塞基础闭环。
输入为最近 `tracking.windowLength` 个已提交观测：单位球面方向、框尺寸、置信度和可用深度范围。

- 方向在单位向量空间平滑，不能直接对 yaw 做线性平均，以避免 `-π/π` 跳变。
- 深度有效率低于 `depth.minValidRatio` 或摘要缺失时，只更新球面方向和尺度，不更新 range 速度。
- 预测输出至少包含 `predictedMotion`、下一帧中心、下一帧 FOV 和恢复搜索半径。
- 运动不确定度先保存在 estimator 私有状态；除非自适应 FOV 无法实现，否则不扩展已有
  `MotionState3D` 公共字段。

低置信时，DTC 保留最近可信状态并生成 `t+1 ... t+K` 的有限预测假设。未来帧出现高置信找回后，
从找回帧重新初始化运动窗口，未提交的未来假设全部丢弃；已经写出的历史结果不得回写。

## 8. 恢复规划

恢复不是在同一帧重复调用 Tracker，而是“当前帧有限 batch + 后续帧预测”的有界流程：

1. `UNCERTAIN`：当前帧扩大上下文并验证 guard/主视图，最多追加一次视图 batch。
2. `RECOVERING`：后续帧按 `ringRadii` 和 `viewsPerRing` 扩大环搜；每轮只请求未覆盖方位。
3. `LOST`：按 `globalSearchInterval` 降频做全景粗搜；其余帧提交预测框并设 `valid=false`。
4. 达到 `maxRecoveryFrames` 仍未找回时，结束当前目标的恢复尝试，不静默重置首帧模板。

同一帧的全部视图一次性送入后端；显存不足时允许确定性分 batch，但候选排序和聚合结果必须与
单 batch 一致。恢复预算始终受 `maxViewsPerFrame` 限制。

## 9. 模板策略

- anchor 模板永久保留。
- `UPDATE_RECENT`：连续稳定帧达到 `stableFramesBeforeUpdate`，候选簇通过运动和尺度一致性后更新。
- `UPDATE_STABLE`：要求更高置信、足够视图支持和深度一致性（若深度有效）。
- `UNCERTAIN`、`RECOVERING`、`LOST` 一律 `KEEP`；找回后至少稳定确认一帧再更新。
- `RESET_TO_ANCHOR` 只在显式恢复失败策略或外部重初始化时使用。

命令继续使用现有 `TemplateCommand` 的 `expectedRevision`，后端负责原子执行和 revision 校验，
DTC 只决定命令，不接触模板特征。

## 10. 与现有接口的对应

| 对象 | DTC 角色 |
|---|---|
| `TrackController.buildInitialization` | 生成首帧模板视图和上下文 |
| `TrackController.commitInitialization` | 初始化后提交第 0 帧 |
| `TrackController.plan` | 生成 guard triplet、主视图和恢复/预测视图 |
| `TrackController.update` | 校验响应、回投影、聚合候选、更新状态并原子提交结果 |
| `SearchPlan` | 携带一帧全部 `ViewSpec`、模板命令和主预测运动 |
| `ProjectedObservation` | 携带单图回投影候选及控制层补充分数 |
| `DepthProcessor` | 后端深度摘要协议；DTC 只消费摘要 |

若未来需要暴露多假设，可在 `SearchPlan` 末尾增加带默认值的只读 `predictionHypotheses`；首个
版本优先由 DTC 私有结构维护，避免破坏已有调用方。

## 11. 实现验收

- 每帧至少三张 guard 视图，且总数不超过 `maxViewsPerFrame`。
- 单图低于 `candidateMinScore` 不污染单帧框；最终框来自候选簇而非简单最高分。
- 目标变大时上下文随初始框和预测框共同扩大；跨经线和 yaw wrap 测试通过。
- 遮挡/消失序列不会卡在同一帧；恢复成功后从找回帧重建运动窗口。
- RGB-D 中深度证据确实影响后端 `fusedScore`；RGB-only 结果保持现有退化契约。
- 模板在不确定、恢复、丢失阶段不更新；revision 错误和乱序响应被拒绝。
- 单线程、四线程以及恢复 batch 分片的逐帧结果一致。

与 DTC 相关的所有可变状态只存在于 T0；任何深度网络、融合头或图像裁剪代码进入 controller，
均视为职责越界。
