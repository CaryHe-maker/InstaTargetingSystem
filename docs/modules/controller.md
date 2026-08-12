# InstaTargetingSystem Controller V2 规范

> 本文描述已经接入运行链路的 V2 控制层。详细设计依据见 `docs/StateMachineV2.md`。
> Controller 只消费 TrackerBackend 的局部观测和 geometry 的回投影结果，不实现 RGB/Depth
> 模型融合，不持有图像或模型特征。

---

## FusedScore 预处理

每次 backend 返回一组 `LocalObservation` 后，应用编排层在投影和调用
`StateEvaluator` 前执行一次单调分段线性对比度拉伸。输入/输出关键点为：

```text
0.00 -> 0.00
0.60 -> 0.10
0.80 -> 0.40
0.90 -> 0.70
0.95 -> 0.90
1.00 -> 1.00
```

关键点之间线性插值。因此原候选排序保持不变，`0.80-0.95` 被显著拉开，`0.60`
及以下被压缩到 `0.00-0.10`。处理会创建新的不可变 `LocalObservation`，后续
`ProjectedObservation`、`DecisionGate`、`StateEvaluator` 和 visualization 均使用新
`fusedScore`，不会再次处理。

## 1. 组件

```text
DepthAwareTrackController (T0 single writer)
  ├─ FrameTransaction        同帧尝试、视图总预算和唯一提交
  ├─ SphericalMotionEstimator
  ├─ RecoveryPlanner
  ├─ StateEvaluator
  ├─ TrackStateMachine       纯 transition reducer
  ├─ TemplatePolicy
  └─ RecoveryMemory
```

| 组件 | 职责 |
|---|---|
| `SphericalMotionEstimator` | 保存最近 `windowLength` 个可靠测量，在球面切平面拟合速度、尺度和可选 range |
| `RecoveryPlanner` | 生成五视图局部覆盖、恢复环和包含南北极的六面 cube-map |
| `StateEvaluator` | 过滤、球面聚类、稳健融合，产生 `StateObservation` |
| `TrackStateMachine` | 根据证据、计数和预算给出下一状态，不拥有跨帧数据 |
| `TemplatePolicy` | 稳定跟踪时发出模板命令，其他状态和找回冷却期保持 `KEEP` |

所有可变状态只由 T0 修改。计划、观测、评估和结果均为不可变跨模块消息。

---

## 2. 状态和证据

内部状态为 `INIT/TRACKING/UNCERTAIN/RECOVERING/LOST/TERMINATED`。公共逐帧结果继续使用
`TrackStatus.TRACKING/UNCERTAIN/RECOVERING/LOST`。

`StateEvaluator` 输出四级证据：

| 证据 | 条件 |
|---|---|
| `CONFIRMED` | 支持视图足够且 `stateScore >= acceptThreshold` |
| `WEAK` | 分数达到 `uncertainThreshold`，但没有达到普通确认条件 |
| `REJECTED` | 无合格候选、低于不确定阈值或硬门控失败 |
| `REACQUIRED` | 恢复/丢失状态中达到更高的 `recoverAcceptThreshold` 和找回支持数 |

主要转移为：

```text
INIT -> TRACKING
TRACKING --weak--> UNCERTAIN
TRACKING --miss--> RECOVERING
UNCERTAIN --confirmed--> TRACKING
UNCERTAIN --patience exhausted--> RECOVERING
RECOVERING --reacquired--> TRACKING
RECOVERING --budget exhausted--> LOST
LOST --candidate--> RECOVERING
LOST --reacquired--> TRACKING
```

同名状态的下一帧会创建新的 `StateInstance/stateId`。`LOST` 不表示同帧重试；同帧重试由
`FrameTransaction.attemptIndex` 表示。

---

## 3. LocalObservation 处理

一个 `LocalObservation` 只属于一个局部视图。应用层先把它回投影为 `ProjectedObservation`，
随后 `StateEvaluator` 执行：

1. 校验 viewId、顺序、分数和框。
2. 使用后端 `fusedScore`、运动、尺度和可用深度证据计算 `decisionScore`。
3. 低于 `candidateMinScore` 的候选只保留诊断意义。
4. 以球面角距离、BFoV 重合和对数尺度差构造候选兼容图。
5. 使用连通分量得到与输入顺序无关的候选簇。
6. 选择一个最佳簇，只融合该簇。
7. 中心使用单位球向量稳健均值，宽高使用加权中位数。

不同区域的候选不会直接并集。互不支持的单图高分候选只能成为恢复搜索种子；支持不足时
最终输出运动预测框且 `valid=false`。候选最大包络不参与 `TrackResult.bbox`，避免放大输出框并降低 IoU。

---

## 4. 多帧运动预测

历史只保存可靠测量：

- `INITIAL`
- `OBSERVED_CONFIRMED`
- `OBSERVED_REACQUIRED`

纯预测输出、弱候选和第一次尝试中被否决的候选不会进入历史。

方向预测在最后可靠中心的局部切平面进行加权常速度拟合，跨经线时不直接相减 yaw。尺度在对数
空间拟合；range 只有在深度有效时才保存，首次有效深度只初始化而不估计速度。预测输出包含角度、
尺度、range、置信度和不确定度。不确定度随时间和缺失增长，用于扩大搜索 FOV，而不是扩大最终目标框。

找回后清除旧历史和未来假设，以找回测量重新初始化，并进入 `reacquireCooldownFrames` 冷却期。

---

## 5. 视图计划

### 5.1 TRACKING/UNCERTAIN

默认使用预测中心主视图和四个切平面角方向保护视图，共五图。`UNCERTAIN` 乘以
`uncertainFovScale` 扩大 FOV。输出像素尺寸固定，FOV 仍受 geometry min/max 限制。

### 5.2 RECOVERING

先验证搜索种子，再按 `ringRadii/viewsPerRing` 生成球面环。`RecoveryMemory.attemptedPlanKeys` 和
`coveredCells` 跨恢复帧保存，避免重复同一区域。

### 5.3 LOST

每 `globalSearchInterval` 帧执行六面 cube-map：四个赤道面和南北极面，面间使用
`cubeMapOverlapRatio` 重叠。非扫描帧使用单个丢失验证视图。

所有尝试合计不超过 `maxViewsPerFrameTotal`。

---

## 6. 帧事务

运行入口使用：

```python
plan = controller.beginFrame(frame)
while True:
    observations = inferAndProject(plan)
    step = controller.consume(plan, observations)
    if isinstance(step, MoreViewsRequired):
        plan = step.plan
        continue
    result = step.result
    break
```

`maxAttemptsPerFrame` 只能为 1 或 2。第一次为 `WEAK/REJECTED` 且预算允许时返回
`MoreViewsRequired`；第二次必须提交。每帧只发布一个 `TrackResult`，已经发布的结果不回写。

旧 `plan/update` 接口仍作为单尝试兼容路径保留，不会返回第二个计划。

Controller revision 每个提交帧增加一次；Backend 模板 revision 每次推理尝试增加一次。两者在同帧
升级后允许不同，不能再要求数值始终相等。

---

## 7. 输出和模板

`TrackResult.resultSource` 表示来源：

- `INITIAL`
- `OBSERVED_CONFIRMED`
- `OBSERVED_REACQUIRED`
- `OBSERVED_WEAK_BLEND`（协议保留，当前正式路径默认不用弱框扩大输出）
- `MOTION_PREDICTED`

只有前面三种可靠测量可以更新运动状态。动态模板还要求稳定 `TRACKING`、足够支持和找回冷却结束。

---

## 8. 失败和原子性

- 旧 transaction/attempt、未知或重复 viewId、乱序响应立即报错。
- 正常非空响应保持请求视图顺序；空序列表示本次模型没有候选。
- 恢复规划使用事务内的 `RecoveryMemory` 副本，只有最终提交才替换跨帧内存。
- 第二次尝试失败不会发布第一次的半成品；运行入口按现有 FatalError 语义终止。
- 状态历史不保存 RGB、Depth、模板特征或无界候选列表。

---

## 9. 验收

- 不相交候选不会生成大并集框。
- 预测输出不回灌运动窗口。
- 每帧最多两次尝试且只有一个结果。
- 六面搜索覆盖赤道、经线和两极。
- 找回后速度窗口重建、模板冷却。
- RGB-only 与 RGB-D 使用同一状态和事务协议。
- 单线程与线程化实现应保持逐帧确定性。
