# 第五阶段修改蓝图：DTC 与控制层建议

> **历史文档，已被取代。** 当前 controller 的实现依据是 [`../StateChangePlan.md`](../StateChangePlan.md) 和 [`../modules/controller.md`](../modules/controller.md)。本文中的 guard triplet、自适应 FOV、环搜及旧视图预算仅用于追溯，不得作为当前修改依据。

> 本文是第五阶段（控制层 DTC）的实现蓝图，不是对现有代码的重写授权。它以当前仓库已经落地的 `core`、`geometry`、`tracker` 和 `visualization` 为基线，吸收 `docs/Suggestions.md` 中的要求，并把必须的接口补充、算法边界和验收条件明确下来。
>
> 优先级约定：**必须**表示没有该项就无法形成可靠的第五阶段闭环；**建议**表示用于提升效果或可维护性；**可选**表示不应阻塞首个可运行版本。

## 1. 基线与总决策

### 1.1 当前基线

- `geometry` 已能完成 ERP/BFoV/局部框之间的球面几何变换，并能把 RGB 与对齐的 Depth 同步裁剪为 `LocalView`。
- `TrackerBackend` 已统一处理 RGB-only 与 RGB-D：深度预处理、深度伪彩色、深度分支和融合头都在后端完成；控制层只应消费 `LocalObservation` 的分数和 `DepthSummary`。
- `LocalObservation` 是**单图结果**：一个 `ViewSpec` 对应一个局部框和一组 `modelScore / appearanceScore / depthScore / fusedScore`。
- `ProjectedObservation` 是把单图结果回投影到球面/ERP 后的候选。第五阶段需要把同一帧的全部候选聚合成一个提交结果。
- 控制器目录中的六个文件目前仍是 TODO 桩，`TrackController` 的规划/提交协议尚未形成运行闭环。
- `visualization` 是旁路诊断模块，不参与跟踪状态或候选排序。

### 1.2 结论（先固定，避免实现中反复摇摆）

1. **Depth-to-RGB 继续放在 TrackerBackend。** 当前方案已经实现，且 geometry 的职责是对齐和裁剪。除非基准测试证明“geometry 预先转换”为可重复的显著吞吐提升，否则不迁移。迁移会引入重复转换、缓存一致性和接口兼容风险。
2. **DTC 不读取整张深度图，也不实现深度编码或 MLP。** 它只消费后端提供的 `depthScore`、`DepthSummary` 和缺失标志，用于门控、尺度估计和恢复排序。
3. **每帧固定包含多视图。** 无论状态是否为 `TRACKING`，都必须包含三张覆盖保护视图（guard triplet）；在此基础上再添加预测中心视图和恢复/尺度所需的自适应视图。后端应以 batch 方式推理。
4. **控制层不把后端融合重做一遍。** `fusedScore` 是模型输出；DTC 只计算一个用于选择/状态转移的轻量 `decisionScore`，并明确记录二者的语义差异。
5. **恢复以“跨后续帧的有界搜索”为中心。** 同一帧最多执行配置的若干视图，不允许卡在一张图上重试；丢失时一边扩大当前帧视野，一边维护未来若干帧的预测假设。

## 2. 术语与输出语义

为避免“单图/单帧/多帧”混用，后续文档和代码统一采用下列定义。这里把 Suggestions.md 中的命名固定为：`LocalObservation.bbox` 是**单图预测框**，`FrameAggregate.bbox` 是**单帧预测框**，`TrackResult.bbox` 在没有可靠观测而由历史状态外推时是**多帧预测框**；`TrackResult.valid` 用来区分真实观测提交和纯预测提交。

| 层级 | 对象 | 含义 |
|---|---|---|
| 单图（per-view） | `LocalObservation` | Tracker 对一个局部视图给出的局部框和分数；不得跨视图平均后再回填此对象 |
| 单帧候选（per-frame candidate） | `ProjectedObservation` | 单图结果回投影后的球面候选，保留 `viewId` 与各项分数 |
| 单帧决策（per-frame aggregate） | DTC 内部 `FrameAggregate`（建议新增） | 当前帧所有候选经阈值过滤、聚类、加权融合后得到的唯一框、置信度和来源 |
| 多帧提交（temporal result） | `TrackResult` | 结合历史窗口后的最终输出；候选可靠时 `valid=true`，仅预测/恢复未确认时 `valid=false` |

`FrameAggregate` 可以先作为 controller 私有数据类；若需要日志或可视化，再以只读结构加入 `core.types`。不应为了表达内部中间量而破坏现有 `TrackResult` 的比赛输出契约。

## 3. 已实现模块的保护范围与必要改动

### 3.1 `geometry`：保持实现，补充契约说明

**不改算法。** 保留现有球面坐标、边界采样、经线跨越处理，以及 RGB/Depth 同视角裁剪。只在 `docs/interface.md` 和 `docs/modules/geometry.md` 明确以下约束：

- `cropViews(frame, specs)` 必须保持 `specs` 顺序、返回同样的 `viewId`，并对每个视图同时裁剪 RGB 和 Depth。
- DTC 生成的每个 `ViewSpec` 必须有唯一 `viewId`，且水平中心角要通过现有 yaw 归一化处理。
- 视图 FOV 不能超出 `GeometryConfig.minFovRad/maxFovRad`；“扩大视野”通过增大 BFoV，而不是修改输出分辨率。
- 局部上下文的最低尺寸按目标框的**宽和高各至少 2 倍**解释，即局部面积至少为首框面积的 4 倍。若受全景边界或 `maxFov` 限制，应记录 `contextClipped=true`，不可静默缩小。
- Depth 缺失时继续返回 `LocalView.depth=None`，不得用零数组伪造深度。

**仅在确有需要时的兼容性增强（建议）。** 若需要在日志中区分 guard/adaptive/recovery 视图，可给 `ViewSpec` 末尾增加带默认值的 `role` 字段；首个版本也可以由 DTC 维护 `viewId -> role` 的私有映射，避免改变已实现的 core 构造方式。

### 3.2 `tracker`：保持后端边界，补测试与文档

- `TrackerBackend.infer()` 继续接收 `Sequence[LocalView]` 并返回同序的 `LocalObservation`；不要把多视图融合移入后端。
- 深度伪彩色图对应的 HiT/编码器输出必须进入 `FusionHead.fuse()`。在 RGB-D 模式下，改变深度分支输入应能改变 `fusedScore`；在 RGB-only 模式下 `depthScore=0` 且行为保持当前退化路径。
- 后端输出的 `fusedScore` 必须经过 `[0,1]` 校验；DTC 不重新计算 RGB/Depth 融合概率。
- `template.Command` 的执行和 revision 校验仍由后端负责，更新时机由 DTC 的 `TemplatePolicy` 决定。

需要补的是契约测试，而不是重写后端：验证深度分支确实参与融合、深度无效时自动退化、批量视图结果顺序稳定、模板更新不会在一次推理中途发生。

### 3.3 `visualization`：旁路保持不变

现有可视化输出、PNG 无损保存和开关语义全部保留。建议只在 `docs/visualization.md` 增加可选的 `dtc_candidates`、`dtc_state`、`dtc_prediction` 诊断阶段；这些数据从 DTC 复制后写出，不得反向影响 `TrackResult` 或实时队列。

## 4. DTC 的推荐架构

DTC 由五个纯职责组件和一个有状态外观组成：

```text
TrackController (T0, single writer)
  ├─ MotionEstimator       窗口运动估计与预测
  ├─ RecoveryPlanner       视图预算、guard triplet、扩窗/环搜/全景搜
  ├─ CandidateAggregator   单帧候选过滤、聚类、融合
  ├─ DecisionGate          decisionScore、阈值与状态转移输入
  └─ TemplatePolicy        KEEP / UPDATE_RECENT / UPDATE_STABLE / RESET_TO_ANCHOR
```

推荐的数据流保持现有线程模型：

```text
Frame(t)
  -> DTC.plan(t) -> geometry.cropViews -> TrackerBackend.infer(batch)
  -> 回投影全部 LocalObservation
  -> CandidateAggregator -> DecisionGate -> DTC.update(t)
  -> TrackResult(t) + 下一帧 SearchPlan
```

DTC 只在 `T0` 线程持有可变状态；`T2` 只持有 Tracker 会话和模板特征。所有计划和响应必须校验 `sequenceId/frameIndex/stateRevision`，旧响应直接报错。

## 5. 每帧视图计划（必须实现）

### 5.1 Guard triplet

每一帧都生成三张覆盖保护视图，中心取“上一帧已提交框的球面中心”或“主预测中心”，yaw 偏移为 `-120°、0°、+120°`，pitch 取当前预测 pitch；FOV 取不小于目标上下文所需值并限制在 geometry 上限。三张图应覆盖左右相邻区域，允许重叠，不得因 `TRACKING` 高置信而省略。

guard triplet 的作用是防止局部视图过拟合、经线附近丢失和短时错误锁定；它们不是三个独立目标，最终仍需经过候选聚类。

### 5.2 主视图与自适应视图

在 guard triplet 之外，按状态添加：

- `TRACKING`：预测中心主视图；必要时加入一个尺度放大视图和一个尺度缩小视图。
- `UNCERTAIN`：以预测中心为中心的更大上下文视图，并加入 2 个角速度/深度变化假设。
- `RECOVERING`：按环搜索生成候选方位；每轮保留上一轮未覆盖的方位，禁止重复同一 `ViewSpec`。
- `LOST`：按 `globalSearchInterval` 降频执行低分辨率全景粗搜；其余帧只输出预测并推进假设。

总数不得超过 `recovery.maxViewsPerFrame`。若预算不足，优先级固定为：`guard triplet > 主预测视图 > 预测假设 > 环搜/全景视图`。

### 5.3 上下文尺寸

对每个主/恢复假设，DTC 计算局部上下文框：

```text
contextWidth  >= 2 * max(initialWidth,  predictedWidth)
contextHeight >= 2 * max(initialHeight, predictedHeight)
```

再叠加 `contextMarginRatio`，并裁剪到 geometry 的可用 FOV。预测框变大时，上下文必须同步变大；不能只沿用上一帧局部图尺寸。目标框过小导致的最小 FOV 由 `minFov` 兜底，目标框过大导致的最大 FOV 由 `maxFov` 兜底。

## 6. 单帧候选聚合

### 6.1 过滤

对每个 `ProjectedObservation` 计算轻量门控分数：

```text
decisionScore = normalize(
    w_backend * fusedScore
  + w_motion  * motionScore
  + w_scale   * scaleScore
  + w_depth   * depthConsistencyScore
)
```

其中 `fusedScore` 是后端已经训练/校准的模型结果；其余项只用于控制层排序。某个模态缺失时，相关权重从分母中移除后重新归一化。`candidateMinScore`（默认建议 0.40）以下的候选不进入聚类和最终框计算，但应保留在诊断日志中。

### 6.2 聚类与融合

1. 将候选的 BFoV 中心转成单位向量，按球面角距离聚类；同簇还需满足尺度比在 `scaleClusterTolerance` 内。
2. 候选之间可用球面 IoU 或“中心角 + 尺度差”作为兼容实现。跨经线框必须使用现有 seam 规则，不得用普通平面 IoU 直接比较。
3. 每簇以 `decisionScore` 为权重，使用加权中位数/截尾均值融合中心、宽高和深度；单个高分离群候选不能把框拉出簇范围。
4. 选择簇时同时考虑总权重、最高分、视图覆盖度和运动连续性。仅来自一张图的高分簇不能自动压过多视图一致的中分簇。
5. 得到的簇输出为“单帧预测框”和“单帧置信度”。该结果随后再与时间窗口融合，形成 `TrackResult` 的多帧提交。

建议先实现确定性的加权几何聚合，再通过离线数据校准权重；不在第五阶段引入新的端到端跨视图神经网络。

## 7. 多帧预测与恢复策略

### 7.1 运动估计

`motion_estimator.py` 先实现可解释的常速度 Alpha-Beta 模型，输入为最近 `windowLength` 个**已提交**观测：单位球面方向、可用深度范围、框尺寸和置信度。球面方向必须在单位向量空间平滑，避免 yaw 在 `-π/π` 处跳变；深度缺失或 `validRatio` 过低时不更新 range 状态。

接口层仍返回现有 `MotionState3D`。估计器内部可以保存角速度、范围速度和不确定度，不强制立即扩展公共类型；只有当自适应 FOV 无法在内部实现时，才以默认字段向 `MotionState3D` 增加预测不确定度。

### 7.2 状态机

状态转移使用双阈值和连续帧计数，避免在阈值附近抖动：

| 当前状态 | 进入条件 | 主要动作 |
|---|---|---|
| `INIT` | 首帧模板初始化成功 | 建立 anchor、motion history，进入 `TRACKING` |
| `TRACKING` | `decisionScore >= acceptThreshold` 且运动/尺度门控通过 | 正常输出，允许按策略更新模板 |
| `TRACKING` | 分数落入 `[uncertainThreshold, acceptThreshold)` 或候选不一致 | 进入 `UNCERTAIN`，扩大上下文，禁止模板更新 |
| `UNCERTAIN` | 连续 `uncertainPatience` 帧未确认 | 进入 `RECOVERING`，启动新预测假设 |
| `UNCERTAIN/RECOVERING` | 任一候选达到 `recoverAcceptThreshold` 并通过跨视图一致性 | 回到 `TRACKING`，从找回帧重置运动窗口 |
| `RECOVERING` | 超过 `maxRecoveryFrames` 或预算耗尽 | 进入 `LOST`，按间隔执行全景粗搜 |
| `LOST` | 全景粗搜连续失败达到上限 | 输出无效预测并结束当前目标；不得静默重置 anchor |

在 `UNCERTAIN/RECOVERING/LOST` 中绝不更新模板。找回后立即丢弃尚未提交的未来假设，从找回帧建立新的时间窗口；已经写出的历史帧不回写，避免破坏“每帧一个结果”的输出顺序。

### 7.3 不阻塞的未来帧假设

当当前帧低置信时，DTC 保留最近可信状态，并生成 `t+1 ... t+K` 的预测中心/FOV 假设。每个后续帧最多消费一次实际图像和一次有界 batch；不能在同一帧上等待“更好的图”。若未来帧出现高置信结果，使用该帧重新初始化 motion history，之前的预测只作为已输出结果和诊断信息。

## 8. 模板策略

- anchor（首帧）永久保留，只能由显式 `RESET_TO_ANCHOR` 清理动态槽。
- recent 模板：连续 `stableFramesBeforeUpdate` 帧通过接收门限，且候选与历史簇一致时更新。
- stable 模板：比 recent 更严格，要求更高置信、深度一致（若有）和足够的跨视图支持。
- `UNCERTAIN`、`RECOVERING`、`LOST` 一律 `KEEP`；找回帧至少再稳定确认一帧后才允许更新。
- 模板命令携带当前 `stateRevision`、`frameIndex`、`viewId` 和局部框；后端按现有 revision 机制原子执行。

## 9. 配置建议

现有字段保持兼容，建议在 `tracking`/`recovery` 中追加以下字段（均提供默认值，旧配置可继续加载）：

| 字段 | 建议初值 | 作用 |
|---|---:|---|
| `tracking.candidateMinScore` | `0.40` | 低于此值的单图候选只记日志，不参与聚合 |
| `tracking.recoverAcceptThreshold` | `0.80` | 找回所需的高置信门限，通常高于普通接收门限 |
| `tracking.uncertainPatience` | `2` | 进入恢复前允许的连续不确定帧数 |
| `tracking.maxRecoveryFrames` | `30` | 单次恢复的最大持续帧数 |
| `tracking.contextScale` | `2.0` | 初始/预测框宽高的最小放大倍数；面积约为 4 倍 |
| `tracking.contextMarginRatio` | `0.15` | 额外上下文边界 |
| `tracking.scaleClusterTolerance` | `0.50` | 候选尺度聚类容差（对数尺度） |
| `tracking.maxPredictionHorizon` | `3` | 未来预测假设的最大帧数 |
| `tracking.guardYawStepDeg` | `120` | guard triplet 的水平中心角步长 |
| `tracking.minViewsForCommit` | `2` | 单帧可提交所需的最少相互支持视图数；初始化帧例外 |
| `recovery.maxViewsPerFrame` | 保留现值 `12`，且必须 `>=3` | 每帧视图预算；三张 guard 视图是硬下限 |
| `recovery.ringRadii` | `[1.0, 1.75, 2.5]` | 恢复环相对上下文半径 |
| `recovery.viewsPerRing` | `[4, 8, 12]` | 各恢复环视图预算，最终受 `maxViewsPerFrame` 限制 |
| `recovery.globalSearchInterval` | 保留现值 `5` | LOST 状态全景粗搜间隔 |

配置加载器必须继续拒绝未知字段；因此新增字段应同步修改 `core/config.py`、YAML 样例和配置单测，不能只在 DTC 中读取任意字典。

## 10. 需要修改的文档与代码落点

以下是实施第五阶段时的最小修改面：

| 优先级 | 文件 | 修改方案 |
|---|---|---|
| 必须 | `docs/interface.md` | 补充单图/单帧/多帧术语、批量视图响应、`valid=false` 预测输出、revision 和候选阈值语义；如采用多假设，再为 `SearchPlan` 增加向后兼容的可选字段 |
| 必须 | `docs/modules/controller.md` | 用本文第 4~8 节替换当前“单视图 RGB-only 预留深度”的描述，写清 DTC 的多视图、聚合、恢复和模板策略 |
| 必须 | `src/instatarget/controller/*.py` | 按组件职责实现 motion、gate、state machine、recovery、template policy 和 facade；不得复制 tracker 的深度融合逻辑 |
| 必须 | `docs/process.md` | 把正常跟踪时的单视图流程改为固定 guard triplet + batch，并写明恢复不重复消费同一帧 |
| 必须 | `docs/implement.md` | 第五阶段清单改为可验证子任务，增加“单帧候选聚合”和“恢复预算/未来假设”两项 |
| 必须 | `docs/hyperparameters.md`、`configs/*.yaml` | 增加第 9 节字段及范围、默认值和 RGB-only 退化规则 |
| 建议 | `docs/data.md` | 明确深度单位、无效掩码、`DepthSummary` 置信度以及缺失模态的表示 |
| 建议 | `docs/visualization.md` | 增加 DTC 候选/状态诊断旁路，默认关闭 |
| 建议 | `docs/modules/tracker.md` | 增加“深度分支必须影响 fusedScore”的契约测试说明，保持实现边界不变 |

不建议在本阶段修改 `geometry` 的投影算法、`tracker/backend.py` 的职责、既有可视化格式或比赛输出格式。若 profiling 证明 geometry 预转换 Depth-to-RGB 明显更快，应另开变更记录，先增加缓存命中/坐标对齐测试，再迁移。

## 11. 实施顺序

1. 先补 `core/config.py` 的新增配置字段和契约单测，确保所有参数有范围检查。
2. 实现 `motion_estimator.py`，用纯球面/合成轨迹测试验证 yaw wrap、深度缺失和尺度变化。
3. 实现 `CandidateAggregator` 与 `decision_gate.py`，先完成确定性过滤、聚类和加权融合，再接状态机。
4. 实现 `recovery_planner.py`，加入 guard triplet、上下文放大、环搜和全景粗搜的视图预算测试。
5. 实现 `state_machine.py` 与 `template_policy.py`，验证连续计数、滞回、找回后重置历史和模板保护。
6. 实现 `depth_aware_track_controller.py` facade，对齐现有 `TrackController` 的 `buildInitialization/commitInitialization/plan/update` 协议。
7. 最后接入 `process/app` 的批量请求，先跑单线程，再验证四线程结果逐帧一致。

## 12. 验收与回归门禁

### 12.1 功能验收

- 每帧至少有三张 guard 视图；任何状态都不会超过 `maxViewsPerFrame`。
- 单图低于 `candidateMinScore` 的结果不会污染单帧框；单帧最终结果来自候选簇而不是简单取最高分。
- 连续遮挡、短时消失和跨经线运动均能从 `TRACKING` 进入恢复并回到 `TRACKING`；同一帧不会无限重试。
- 找回后从找回帧重建运动窗口，后续预测不沿用失效的未来假设。
- `rgb_depth` 下改变深度证据会影响后端 `fusedScore`；`rgb_only` 下输出与现有基线一致。
- 模板在不确定/恢复/丢失阶段不更新，revision 错误、乱序响应和重复 viewId 会被拒绝。

### 12.2 质量与性能指标

除现有 AUC、Success、FPS 外，至少记录：恢复延迟（帧数）、误找回率、连续丢失长度、单帧平均视图数、guard 命中率、深度有效率、RGB-D 相对 RGB-only 的收益和额外延迟。

### 12.3 必测场景

1. 目标平滑运动、快速转向、yaw 从 `+π` 跨到 `-π`。
2. 目标逐渐变大/变小，验证上下文随初始框和预测框共同扩大。
3. 目标被遮挡 1~N 帧后重新出现，验证未来假设、恢复预算和历史重置。
4. 深度部分缺失、全缺失、跳变异常，验证权重归一化和 RGB-only 退化。
5. 多视图中存在高分离群框，验证球面聚类和最小支持视图数。
6. 单线程/四线程、不同 batch 分片，验证结果顺序和候选排序一致。

## 13. 最终建议

第五阶段的最小可交付版本应先实现“固定三视图保护 + 自适应上下文 + 确定性单帧聚合 + 有界恢复状态机”，再考虑 Kalman、学习式候选校准或端到端多视图模型。这样可以最大限度保持已完成的 geometry、tracker、visualization，且把 Suggestions.md 中真正影响跟踪效果的要求落实到可测试的控制层行为上。
