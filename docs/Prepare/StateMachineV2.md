# InstaTargetingSystem 状态机与控制器重构蓝图 V2

> **历史文档，已停止作为实现依据。** 当前已实现的固定 120° 视域、按状态区分的 2/3/3/2 轮路线、双框 FuseBox、四级测量证据及状态转换规则，以 [`../StateChangePlan.md`](../StateChangePlan.md) 为唯一执行规范。本文保留用于追溯早期设计，其中的自适应 FOV、五视图、恢复环搜和旧轮次规则不得用于修改当前实现。

> 本文最初是 `StateMachine.md` 的评审和 Controller V2 重构蓝图，以下内容记录当时方案。
> 2026-08-11 已落地：状态实例、帧事务、`StateEvaluator`、可靠测量滑动窗口、状态相关五视图、
> 恢复记忆、六面 cube-map、同帧最多一次升级、结果来源和独立 backend revision。
> 本文中的“当前缺口/后续修改”章节保留设计审计价值；实际交付状态以第 13 节标记为准。
>
> 本文保留原文的重要意图：一个目标任务只有一个状态机；同类型状态的每次进入都是不同状态实例；
> 每个状态实例拥有自己的预测中心和视图计划；多视图观测由 `StateEvaluator` 统一变成
> `StateObservation`；运动预测使用历史结果中心而不是把预测中心当测量；低置信时可扩大视野并找回；
> 同一帧不得因反复进入 `LOST` 而卡死。

---

## 1. 文档结论

### 1.1 对原 `StateMachine.md` 的判断

原设计相对当前实现有真实的优化价值，主要体现在：

1. 明确提出“每次进入状态都是唯一实例”，比当前只有枚举和计数器更适合诊断、回放和原子提交。
2. 用 `StateEvaluator -> StateObservation` 收口多视图融合，能替代当前职责偏薄的
   `FrameAggregate + StateUpdate` 两段式信息丢失。
3. 区分 `PredictedCenter` 与本帧结果中心，明确禁止把预测值当作新的运动测量，方向正确。
4. 低分后在同一帧扩大搜索再覆盖本帧候选，有机会减少“明明本帧还能找回，却只能等下一帧”的延迟。
5. `TRACKING/UNCERTAIN` 的五视图和 `RECOVERING/LOST` 的六面搜索表达了状态相关覆盖策略，
   比所有状态使用相同视图模板更有针对性。

但原设计不能原样实现。以下问题必须先修正：

- `lost[n] -> tracking[n]/uncertain[n]` 把持久状态和同帧推理阶段混在一起，必须改成有界的
  `PRIMARY -> ESCALATION` 帧内尝试；`LOST` 只表示跨帧的长期丢失状态。
- `EverLost` 是防循环补丁。V2 用 `attemptIndex`、`maxAttemptsPerFrame` 和帧事务预算从结构上保证终止，
  不保留该布尔字段。
- “返回原状态对象并更新它”违背项目跨模块不可变对象与原子提交规范。V2 每次生成新的不可变状态实例。
- 原文的 `tracking -> output`、`recovery -> output` 把输出当状态转移。V2 规定每个输入帧最终恰好提交一次
  `TrackResult`，EOF 再进入 `TERMINATED`，输出不是状态。
- `score`、`StateScore`、相交和并集框算法定义不足。直接取多个框的最大并集会随离群框膨胀，不能作为最终框。
- 所有视图始终使用最大 FOV 会显著降低目标像素占比。V2 保留“局部图输出分辨率固定且充分大”，
  但 FOV 按状态、目标尺度和预测不确定度自适应；只有全局扫描接近最大 FOV。
- 恢复成功不能只看单一阈值，还必须验证锚点外观、多视图支持、球面位置、尺度和可用深度一致性。

因此，V2 的总体决定是：保留原文的状态实例、显式评估器、结果中心语义、五/六视图覆盖意图和同帧找回能力，
但以“帧事务 + 有界尝试 + 纯状态转移 + 滑动窗口运动模型”重新组织。

### 1.2 当前代码已经具备的基础

以下能力应复用，不应在重构中推倒重来：

- `geometry` 已处理 ERP/BFoV/局部框、经线循环和 RGB/Depth 同步裁剪。
- `TrackerBackend` 已统一输出 `LocalObservation`，并在后端内部负责 RGB-D 融合。
- `DecisionGate` 已具备候选过滤、球面聚类和轻量控制分数的雏形。
- `RecoveryPlanner` 已具备 guard、尺度、环搜和视图预算的雏形。
- `TemplatePolicy` 已保护不确定、恢复和丢失阶段的模板。
- `DepthAwareTrackController` 已校验帧顺序和 revision，并具有 plan/update 闭环。
- 公共消息大多已经是 `frozen=True, slots=True` 的不可变数据类。

### 1.3 当前实现与目标文档之间的实际缺口

| 当前事实 | 风险 | V2 目标 |
|---|---|---|
| `windowLength` 只被校验，未被运动算法消费 | 名义多帧、实际单状态滤波 | 显式保存最近 `n` 个可靠测量并做球面窗口拟合 |
| `uncertainThreshold` 未参与当前状态机判断 | 三段分数只剩一个普通接收门限 | 明确 `REJECTED/WEAK/CONFIRMED/REACQUIRED` 四级证据 |
| `maxPredictionHorizon` 未产生多步假设 | 遮挡期搜索半径没有真实预测依据 | 生成有置信衰减和不确定度增长的有限预测 |
| 当前 Alpha-Beta 只有一个位置/速度状态 | 无法做离群剔除、窗口重拟合和残差估计 | 窗口加权稳健拟合，Alpha-Beta 作为样本不足时的降级 |
| 无深度初始化时 `rangeDepth=0` | 后续首次深度可能产生错误的大速度 | range 使用 `None/valid` 语义，首次有效深度只初始化不估速 |
| 找回后直接用旧运动状态更新 | 旧速度会污染新轨迹 | 找回帧重建运动窗口并丢弃旧未来假设 |
| `StateUpdate` 仅有状态、计数和两个布尔值 | 无法解释转移原因和结果来源 | 完整 `StateObservation` 与 `TransitionDecision` |
| 状态机在 controller 完成全部提交前已原地修改 | 后续运动更新异常时无法保证原状态不变 | 纯 reducer + copy-on-write 帧事务，一次原子提交 |
| 候选预过滤使用 `fusedScore`，聚合后才有控制分数 | 运动/尺度明显不合理的候选仍可入簇 | 先算完整候选证据，再按配置过滤与聚类 |
| 恢复环的去重集合只存在于一次 `buildViews()` | 后续帧会重复搜索相同方位 | `RecoveryMemory.coveredCells` 跨状态实例保存 |
| 当前全局视图仅沿赤道排列 | 极区不能被真正全局覆盖 | 四个赤道面 + 南北极面的六面 cube-map |
| `LOST` 无终止/外部重初始化语义 | 可永久增长计数并无限运行 | 显式低频扫描、终止策略和可选外部重初始化 |
| 观测校验允许响应为请求视图子集 | 与后端“一一对应”契约冲突 | 正常响应严格一一对应；部分响应只能是显式降级事件 |

---

## 2. V2 不可破坏的不变量

1. 一个连续单目标任务只有一个 `TrackSessionController` 和一个状态机所有者。
2. 所有可变控制状态只允许 T0 控制线程写入；T2 只持有模型会话和模板特征。
3. 输入帧严格递增。每个输入帧最终恰好产生一个 `TrackResult`，结果一经发布不回写。
4. 每帧最多有 `maxAttemptsPerFrame` 次推理尝试，建议默认 2；总视图数和总时延同时受预算限制。
5. 同一帧的多次尝试属于一个 `FrameTransaction`，不是多个持久 `LOST/TRACKING` 状态。
6. 每个状态实例、计划、响应、评估和提交都携带 `sequenceId/frameIndex/stateRevision/transactionId`。
7. `LocalObservation` 是单图局部观测，`ProjectedObservation` 是回投影候选，
   `StateObservation` 是单次状态评估，`TrackResult` 是最终逐帧输出，四者不得混用。
8. `fusedScore` 始终属于 TrackerBackend；控制层只能计算 `decisionScore/stateScore`，不得覆盖它。
9. 球面连续性一律用单位向量、球面距离或切平面计算，不直接跨 `-π/+π` 对 yaw 做线性差分。
10. 运动模型只把可靠的观测中心作为测量。预测中心、纯预测输出和未确认找回候选不得回灌为测量。
11. 模板只在稳定 `TRACKING` 中更新；`UNCERTAIN/RECOVERING/LOST` 和找回冷却期一律 `KEEP`。
12. 状态和历史不保存 ERP、局部 RGB、Depth 数组或模型特征；大数据只在帧事务/后端生命周期内引用。
13. RGB-only 与 RGB-D 使用同一状态机。深度缺失只使深度证据不可用，不改变状态协议。
14. 任何异常、旧 revision、重复提交或预算超限都不能留下半提交状态。

---

## 3. 术语与中心语义

原文关于 `PredictedCenter` 和 `ResultCenter` 的注释非常重要，但“结果中心”需要进一步拆分，
否则纯预测结果可能被错误地当成下一次测量。

| 名称 | 定义 | 是否可更新运动历史 |
|---|---|---|
| `predictedCenter` | 处理本帧前，仅由此前可靠历史预测出的先验中心 | 否 |
| `measuredCenter` | `StateEvaluator` 从本帧候选簇融合出的测量中心 | 只有提交为可靠观测后才可以 |
| `outputCenter` | 本帧最终写入 `TrackResult` 的中心，可能来自测量、预测或二者保守融合 | 否，不能仅因输出就更新 |
| `searchSeedCenter` | 生成下一尝试视图的中心；同帧找回时可采用最优未确认候选 | 否 |
| `lastConfirmedCenter` | 最近一次可靠、已原子提交的测量中心 | 已经在历史中 |

必须保留并强化原文规则：多帧预测基于前几帧可靠的结果测量中心，而不是基于历史
`predictedCenter`。如果某帧 `TrackResult.valid=false`，其 `outputCenter` 只是比赛格式所需的占位预测，
不能加入运动测量窗口。

---

## 4. 状态、状态实例与帧内尝试

### 4.1 持久状态枚举

V2 使用以下内部状态：

```python
class TrackMode(Enum):
    INIT = auto()
    TRACKING = auto()
    UNCERTAIN = auto()
    RECOVERING = auto()
    LOST = auto()
    TERMINATED = auto()
```

- `INIT`：只处理第 0 帧初始化，成功后进入 `TRACKING`。
- `TRACKING`：最近证据可靠，局部预测和模板策略均可正常工作。
- `UNCERTAIN`：仍有目标证据但不足以确认，扩大局部覆盖，不更新模板。
- `RECOVERING`：主动、有界地执行预测假设、环搜和全局覆盖，不更新模板。
- `LOST`：主动恢复预算已经耗尽，按较低频率做全局重识别，其余帧只推进预测。
- `TERMINATED`：EOF、外部取消、不可恢复错误或明确放弃后的终态。

公共 `TrackStatus` 可以暂时继续只暴露 `TRACKING/UNCERTAIN/RECOVERING/LOST`；
`INIT` 以第 0 帧的 `TRACKING` 结果兼容现有输出，`TERMINATED` 不产生额外比赛结果。

### 4.2 每次进入都是不同状态实例

保留原文“`tracking -> tracking` 的两个 tracking 也是不同 state”的要求。V2 不为每个枚举写一个
可变子类，而是使用统一的不可变实例：

```python
@dataclass(frozen=True, slots=True)
class StateInstance:
    stateId: int
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    mode: TrackMode
    enteredFrom: TrackMode | None
    entryReason: TransitionReason
    prediction: MotionPrediction | None
    searchSeedCenter: SphericalPoint | None
    recoveryEpochId: int | None
    modeAgeFrames: int
    stableStreak: int
    weakStreak: int
    missStreak: int
```

`TRACKING[n] -> TRACKING[n+1]` 会创建新的 `stateId` 和新的 `StateInstance`。实例不在完成后修改；
本帧计划、观测和转移写入独立的 `StateRecord`。

### 4.3 帧内尝试不是持久状态

原文的 `lost[n] -> tracking[n]` 真实意图是“同一帧第一次低分后，用更大范围再查一次并覆盖本帧结果”。
V2 将其表示为：

```python
class AttemptKind(Enum):
    PRIMARY = auto()
    ESCALATION = auto()


@dataclass(slots=True)
class FrameTransaction:
    transactionId: int
    frame: FramePacket
    state: StateInstance
    attemptIndex: int
    escalationUsed: bool
    remainingViews: int
    deadlineNs: int | None
    attemptRecords: list[AttemptRecord]
```

规则如下：

1. `PRIMARY` 低分时，`StateEvaluator` 可以返回 `ESCALATE`，但此时不发布结果、不改变持久状态。
2. `ESCALATION` 使用第一次评估给出的 `searchSeedCenter` 或运动预测中心生成更广视图。
3. `attemptIndex` 从 0 开始且严格递增；达到 `maxAttemptsPerFrame`、视图预算或时延预算后必须提交。
4. 同帧只允许一个活动事务；提交后清空全部局部观测引用。
5. `EverLost` 不再存在。`escalationUsed` 只活在当前帧事务中，自然避免同帧死循环。
6. 如果实时配置禁用同帧升级，第一次评估直接提交，下一帧按新状态扩大搜索。

这保留了原设计的即时找回能力，同时维持“每帧只发布一次、已发布结果不覆盖”的输出契约。

---

## 5. 完整状态转移

### 5.1 证据等级

状态机不直接解释一堆原始分数，只消费 `StateEvaluator` 给出的证据等级：

| 等级 | 基本条件 | 含义 |
|---|---|---|
| `CONFIRMED` | `stateScore >= acceptThreshold`，通过硬门控并满足普通多视图支持 | 可靠跟踪测量 |
| `WEAK` | `uncertainThreshold <= stateScore < acceptThreshold`，或分数足够但支持不足 | 可用于搜索提示，不可更新运动/模板 |
| `REJECTED` | 无合格候选、`stateScore < uncertainThreshold` 或硬门控失败 | 本帧没有可信测量 |
| `REACQUIRED` | `stateScore >= recoverAcceptThreshold`，锚点外观、支持数和恢复门控全部通过 | 严格找回测量 |

阈值边界统一使用 `>=`，避免原文中 `>` 与 `>=` 混用造成恰好等于阈值时无转移。

### 5.2 转移表

| 当前状态 | 证据/条件 | 帧内动作 | 提交结果 | 下一状态 |
|---|---|---|---|---|
| `INIT` | 后端模板初始化成功 | 无预测 | 初始框，`valid=true` | `TRACKING` |
| `INIT` | 初始化失败 | 终止并保留错误 | 不发布伪结果 | `TERMINATED` |
| `TRACKING` | `CONFIRMED` | 无升级 | 观测框，`valid=true` | `TRACKING` |
| `TRACKING` | `WEAK` | 有预算可升级一次 | 以弱候选/预测为新搜索种子 | 升级后再决定；否则 `UNCERTAIN` |
| `TRACKING` | `REJECTED` | 有预算可升级一次 | 纯预测或升级结果 | 升级失败后 `RECOVERING` |
| `UNCERTAIN` | `CONFIRMED` 且满足退出滞回 | 无升级 | 观测框，`valid=true` | `TRACKING` |
| `UNCERTAIN` | `WEAK` 且未到 `uncertainPatience` | 可选升级 | 保守融合/预测，`valid=false` | `UNCERTAIN` |
| `UNCERTAIN` | `WEAK` 达到 patience 或 `REJECTED` | 升级或下一帧主动搜索 | 预测，`valid=false` | `RECOVERING` |
| `RECOVERING` | `REACQUIRED` | 清空旧未来假设 | 找回框，`valid=true` | `TRACKING` |
| `RECOVERING` | `CONFIRMED` 但未满足严格找回门控 | 保留为验证种子 | 保守结果，`valid=false` | `UNCERTAIN` |
| `RECOVERING` | `WEAK/REJECTED` 且预算未耗尽 | 推进搜索游标 | 预测，`valid=false` | `RECOVERING` |
| `RECOVERING` | 超过 `maxRecoveryFrames` 或覆盖预算耗尽 | 结束本次恢复 epoch | 预测，`valid=false` | `LOST` |
| `LOST` | `REACQUIRED` | 重建运动窗口 | 找回框，`valid=true` | `TRACKING` |
| `LOST` | `CONFIRMED/WEAK` 但未严格找回 | 开启新验证 epoch | 保守结果，`valid=false` | `RECOVERING` |
| `LOST` | `REJECTED` | 按间隔推进 cube-map 游标 | 预测，`valid=false` | `LOST` |
| 任意活动状态 | EOF | 完成最后一帧后终止 | 不新增帧 | `TERMINATED` |
| 任意活动状态 | 外部显式重初始化 | 新建 session 或显式 reset 事务 | 按外部协议 | `INIT` |

### 5.3 滞回、计数和找回冷却

- `stableStreak`：连续可靠 `TRACKING` 帧数，只在可靠观测提交后增加。
- `weakStreak`：连续弱证据帧数，可靠提交后清零。
- `missStreak`：连续拒绝帧数，可靠提交后清零。
- `modeAgeFrames`：当前模式已持续的提交帧数。
- `reacquireCooldownFrames`：找回后冻结动态模板和速度外推的帧数。
- `recoveryEpochId`：每次从不确定/丢失发起的新恢复阶段递增，防止旧响应或旧搜索游标复用。

这些计数属于跨状态控制内存；`StateInstance` 只保存进入本帧时的快照。

---

## 6. 每个状态的视图计划

### 6.1 对原五视图/六视图注释的保留与修正

原文重要要求如下：

- `TRACKING/UNCERTAIN` 使用 5 张图：预测中心图 + 以该视场四角为中心的 4 张相邻图。
- `RECOVERY/LOST` 使用 cube-map 布置的 6 张图。
- 局部图应尽量大，避免只看见目标局部并误跟；原文建议每张使用最大视场。

V2 保留前两项作为覆盖模板，但把“所有图固定最大 FOV”改为“固定输出分辨率 + 自适应 FOV”：

- 固定最大 FOV 会让稳定跟踪时目标在 256×256 图中占比过小，降低 HiT 定位精度。
- 局部视图的宽高仍使用 geometry 配置的完整输出尺寸，不做低分辨率随意缩图。
- 主视图 FOV 至少覆盖 `contextScale * max(anchorSize, confirmedSize)`，再加预测不确定度边界。
- 相邻图和恢复图可以使用更大 FOV；真正全局 cube-map 才接近 `maxFov`。

### 6.2 视图角色

```python
class ViewRole(Enum):
    PRIMARY = auto()
    LOCAL_CORNER_GUARD = auto()
    MOTION_HYPOTHESIS = auto()
    SCALE_WIDE = auto()
    RECOVERY_RING = auto()
    CUBEMAP_EQUATOR = auto()
    CUBEMAP_POLE = auto()
```

`ViewSpec` 与 `LocalObservation` 仍按 `viewId` 一一对应。建议给 `ViewSpec` 增加可选 `role`，
或由 `SearchPlan.viewRoles: dict[int, ViewRole]` 携带；不得只靠 viewId 数值猜角色。

### 6.3 状态策略

#### `TRACKING`

- 默认 5 图：预测中心主图 + 主图四个切平面角方向的重叠保护图。
- 若预算紧张，可配置成主图 + 两个主要运动方向保护图，但不得少于 `minTrackingViews`。
- FOV 使用目标上下文和预测角不确定度；不执行全景最大视场。
- 尺度快速变化时，可用 `SCALE_WIDE` 替换一个重复度最高的角图，不无上限加图。

#### `UNCERTAIN`

- 基础仍为 5 图，但 FOV 乘 `uncertainFovScale`。
- 四个角图可按预测速度方向旋转，使更多预算落在运动前方。
- 可以加入 `t+1..t+K` 中不确定度最大的两个假设，但总数受本帧预算限制。

#### `RECOVERING`

- 优先从最近可靠中心和最佳未确认候选之间生成局部球面环。
- 使用 `RecoveryMemory.coveredCells` 跳过已搜索且最近没有证据的球面单元。
- 局部环失败或预测不确定度覆盖大半球时，升级为六面 cube-map。
- cube-map 四个赤道面中心的 yaw 间隔 90°；另有 pitch 接近 +90°/-90° 的南北极面；
  各面带少量重叠，避免面边界漏检。

#### `LOST`

- 每 `globalSearchInterval` 帧执行一次旋转后的六面 cube-map；旋转相位跨扫描帧变化以覆盖接缝。
- 非扫描帧可以只生成一张预测/最佳种子验证图，或在严格实时配置下跳过后端并输出预测。
- 高分候选先进入 `RECOVERING` 验证，不因单图高分直接污染模板。

### 6.4 同帧升级视图

第一次尝试为 `WEAK/REJECTED` 时：

1. 有可靠弱候选：以其 `measuredCenter` 作为 `searchSeedCenter`，扩大 FOV 并补相邻/环视图。
2. 无候选：以 `predictedCenter` 为中心，使用不确定度椭圆的长轴方向生成视图。
3. 当前已在 `RECOVERING/LOST`：推进恢复游标，不重复第一次尝试的 `ViewSpec`。
4. 第二次尝试的 viewId 在同一事务内仍全局唯一。
5. 两次尝试合计不超过 `maxViewsPerFrameTotal`，不是每次各享有一份完整预算。

---

## 7. `StateEvaluator` 完整设计

### 7.1 职责

`StateEvaluator` 是纯计算组件，负责把一次尝试的 `ProjectedObservation` 列表转换成唯一的
`StateObservation`。它负责证据规范化、候选过滤、球面聚类、稳健融合、状态分数和结果提案；
它不修改控制内存、不做状态转移、不决定模板命令，也不执行后端 RGB/Depth 融合。

当前 `DecisionGate.score()/aggregate()` 的有效逻辑应迁入或成为它的内部依赖。

### 7.2 输入

```python
@dataclass(frozen=True, slots=True)
class EvaluationInput:
    state: StateInstance
    plan: SearchPlan
    prediction: MotionPrediction
    observations: tuple[ProjectedObservation, ...]
    lastConfirmed: ConfirmedTargetState
    recoveryMemory: RecoveryMemoryView
    config: EvaluatorConfig
```

输入必须包含视图角色、预测不确定度、上次可靠尺度/深度和本次恢复 epoch，不能只给裸观测列表。

### 7.3 输出 `StateObservation`

```python
@dataclass(frozen=True, slots=True)
class StateObservation:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    transactionId: int
    stateId: int
    attemptIndex: int
    evaluatedMode: TrackMode

    predictedCenter: SphericalPoint
    searchSeedCenter: SphericalPoint
    measuredBfov: BFoV | None
    measuredBbox: BBoxXYWH | None
    measuredCenter: SphericalPoint | None
    proposedOutputBfov: BFoV
    proposedOutputBbox: BBoxXYWH
    proposedResultSource: ResultSource

    candidateCount: int
    eligibleCandidateCount: int
    clusterCount: int
    sourceViewIds: tuple[int, ...]
    representativeViewId: int | None
    supportViewCount: int

    backendScore: float
    motionScore: float
    scaleScore: float
    depthConsistencyScore: float | None
    supportScore: float
    agreementScore: float
    stateScore: float

    evidence: EvidenceLevel
    hardGatePassed: bool
    supported: bool
    escalationRecommended: bool
    reacquired: bool
    depthSummary: DepthSummary | None
    rejectionReasons: tuple[EvaluationReason, ...]
```

原文的 `StateObservation.state` 对应 `evaluatedMode`；原文的预测框、`StateScore` 和
`ResultCenter` 分别由 `measured/proposedOutput`、`stateScore`、`measuredCenter/outputCenter` 明确表达。
`EverLost` 被帧事务字段替代，不属于 `StateObservation`。

### 7.4 处理顺序

#### 第 1 步：协议校验

- 校验 sequence、frame、revision、transaction、attempt 和 recovery epoch。
- 正常推理响应必须与请求视图数量、顺序、viewId 一一对应。
- 非有限框、分数越界、未知 viewId、重复 viewId 立即抛 `ProtocolError`。
- 若允许设备部分失败，必须以显式 `PartialInferResponse` 表示；不能把缺失观测伪装成正常空列表。

#### 第 2 步：候选证据

每个候选保留以下分量：

- `backendScore = fusedScore`，不重新融合 RGB/Depth。
- `motionScore`：候选中心相对预测分布的球面残差分数，不只用固定余弦相似度。
- `scaleScore`：候选对数角宽/角高相对预测尺度分布的残差分数。
- `depthConsistencyScore`：仅在深度摘要有效且历史 range 有效时计算；无深度为 `None` 而不是 0。
- `viewQualityScore`：候选是否贴近局部图边缘、是否被裁切、该视图是否接近极区畸变边界。
- `rolePrior`：用于同分排序和诊断，不允许压过明显更强的真实模型证据。

先把有效权重重新归一化，再得到候选 `decisionScore`。过滤应使用完整 `decisionScore` 与必要硬门控，
不再只看 `fusedScore`。

#### 第 3 步：球面聚类

1. 将中心变为单位向量。
2. 以球面角距离、BFoV 球面重合度和对数尺度差建立候选图。
3. 图的连通分量形成簇；禁止按第一个候选贪心归类导致顺序依赖。
4. 同一物理视图的重复候选最多贡献一次支持计数。
5. 跨经线比较必须使用 BFoV/球面表示，不直接比较展开 ERP 的普通矩形。

#### 第 4 步：簇融合

原文注释“相交时选择最大并集矩形；不相交时选最高置信候选”保留其意图，但修改实现：

- 相交候选的最大并集只保存为 `supportEnvelope` 诊断字段，不作为最终框，避免离群框把结果无限放大。
- 最终中心使用单位向量的加权球面中值或稳健归一化均值。
- 宽、高在对数空间使用加权中位数/截尾均值。
- 深度只融合有效摘要，按置信度和有效率加权。
- 没有相交簇时，最高分候选可以成为 `searchSeedCenter` 和弱输出提案；若没有满足
  `minViewsForCommit`，不能直接成为 `CONFIRMED/REACQUIRED`。

#### 第 5 步：状态分数

建议首版使用可解释、可校准的确定性公式：

```text
evidenceMean = normalizedWeightedMean(
    backendScore,
    motionScore,
    scaleScore,
    optional depthConsistencyScore,
    viewQualityScore,
)

supportScore   = min(1, distinctSupportingViews / requiredSupportingViews)
agreementScore = exp(-angularDispersion / angularScale)
               * exp(-logScaleDispersion / scaleTolerance)

stateScore = clamp(
    evidenceMean
    * (baseAgreementWeight
       + supportWeight * supportScore
       + agreementWeight * agreementScore)
    - hardPenalty,
    0,
    1,
)
```

要求：三个组合权重之和为 1；模态缺失时重新归一化；所有阈值和权重登记配置；
`hardPenalty` 只来自明确的裁切、深度跳变、锚点外观冲突等事件，不写隐藏魔数。

#### 第 6 步：证据分类和结果提案

- `CONFIRMED/REACQUIRED`：提案为测量框，允许状态机决定提交为 `valid=true`。
- `WEAK` 且门控基本通过：输出可以是预测与测量的保守球面插值，但 `valid=false`，且不更新运动历史。
- `REJECTED`：提案为纯预测框；预测不确定度只扩大搜索 FOV，不应直接把输出框扩大到整个搜索区。
- 找回判断使用比普通跟踪更严格的 `recoverAcceptThreshold`、锚点外观和最小支持视图数。

---

## 8. 多帧运动预测模块

### 8.1 目标

模块必须真正消费最近 `tracking.windowLength` 个可靠测量，输出下一帧/有限未来帧的球面中心、
角速度、目标角尺度、可选距离和不确定度。它不做完整三维场景重建，也不读取原始深度图。

### 8.2 历史样本

```python
@dataclass(frozen=True, slots=True)
class MotionSample:
    frameIndex: FrameIndex
    timestampNs: int
    center: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    confidence: float
    rangeDepth: float | None
    rangeConfidence: float
    source: MeasurementSource  # INITIAL / CONFIRMED / REACQUIRED
```

只保存可靠的 `measuredCenter`。以下内容不得进入该 deque：

- `predictedCenter`；
- `valid=false` 的 `TrackResult.outputCenter`；
- 单图候选；
- 未通过严格门控的 recovery/lost 候选；
- 同帧第一次尝试被第二次尝试否决的测量。

窗口最大长度为 `windowLength`，但恢复所需的最后可靠锚点可另外保存在 `lastConfirmed`，不靠扩大 deque。

### 8.3 球面方向拟合

推荐首版使用“球面切平面稳健常速度”，Alpha-Beta 作为样本不足时的降级：

1. 取最近可靠中心 `u0` 作为切平面原点，构造正交基 `e1/e2`。
2. 用球面 log map 把窗口中每个单位向量映射为二维切向位移 `q_i`。
3. 以时间差、观测置信度和样本新鲜度为权重，对 `q_i = q0 + v * dt` 做加权最小二乘。
4. 使用一次 Huber/截尾残差重加权，减小单帧错误测量对速度的影响。
5. 通过 sphere exp map 把 `v * horizon` 映射回单位球面得到预测中心。
6. 当切平面跨度超过配置上限、样本少于 2 个或时间间隔异常时，退化为零速度或 Alpha-Beta。

这样可自然处理经线跨越，也比直接对三维向量做线性外推再归一化更容易估计角残差。

### 8.4 尺度与深度拟合

- 对 `horizontalSizeRad/verticalSizeRad` 的对数做加权常速度拟合，并限制每帧最大变化率。
- 距离在 `log(rangeDepth)` 空间拟合；只有 `validRatio/confidence` 达标的摘要才能形成样本。
- 没有历史深度时，range 保持 `None`，不能用 0 代替。
- 首次出现有效深度时只初始化 range，不估计 range 速度；至少两次有效深度后才估速。
- 深度长期缺失时 range 不确定度增长，但方向和尺度预测继续工作。

### 8.5 不确定度

`MotionPrediction` 必须显式携带不确定度，不能只返回一个点：

```python
@dataclass(frozen=True, slots=True)
class MotionPrediction:
    sourceRevision: int
    targetFrameIndex: FrameIndex
    horizonFrames: int
    center: SphericalPoint
    tangentVelocityRadPerSec: tuple[float, float]
    horizontalSizeRad: float
    verticalSizeRad: float
    rangeDepth: float | None
    rangeVelocityPerSec: float | None
    angularUncertaintyRad: float
    scaleUncertainty: float
    rangeUncertainty: float | None
    confidence: float
    degradedReasons: tuple[PredictionDegradedReason, ...]
```

不确定度来自窗口残差、样本数、时间间隔、预测 horizon 和状态模式：

```text
sigmaAngle(h) = clamp(
    residualSigma
    + processNoise * deltaTime
    + missingMeasurementGrowth * missStreak,
    minSigma,
    maxSigma,
)
```

`confidence` 随 horizon 和连续缺失衰减；当前实现“无论预测多久置信度不变”的行为必须修正。

### 8.6 多步假设

- 正常帧至少产生 `t+1` 预测。
- `UNCERTAIN/RECOVERING/LOST` 可生成至 `maxPredictionHorizon` 的预测序列。
- 这些未来预测只是搜索假设，不提前生成或写出未来 `TrackResult`。
- 每次出现可靠找回，旧假设全部失效；以找回帧重新建立 history 和 sourceRevision。
- 视图规划按不确定度选择少量代表假设，不能把 K 个预测全部无条件转为视图。

### 8.7 找回后的重建

找回时不得把新中心直接喂给带旧速度的滤波器：

1. 清空旧运动窗口和未消费未来假设。
2. 以找回测量建立第一个新样本。
3. 若丢失间隔很短且球面残差在安全范围内，可保留最后可靠锚点作为低权重第二样本；否则速度从 0 开始。
4. 进入 `reacquireCooldownFrames`，期间扩大 FOV、禁止模板更新，并限制速度上限。

### 8.8 原文 Lost 返回中心的正确落点

原文要求“从 Lost 返回时，将 Lost 的 ResultCenter 作为 PredictedCenter”。V2 保留搜索意义，
但不把它当运动测量：最佳 lost 候选写入 `RecoveryMemory.bestSeedCenter`，下一次计划将其作为
`searchSeedCenter`；只有严格 `REACQUIRED` 后，它才成为新的 `MotionSample.center`。

---

## 9. 状态机数据保存说明

### 9.1 数据分层

| 层 | 生命周期 | 可变性/所有者 | 保存内容 |
|---|---|---|---|
| `AppConfig` | 整个进程 | 只读 | 阈值、窗口、预算、视图和模板配置 |
| `ControllerMemory` | 一个目标 session | T0 单写 | 跨状态可靠历史、恢复/模板上下文、revision |
| `StateInstance` | 一帧的状态进入 | 不可变 | 模式、预测、进入原因和计数快照 |
| `FrameTransaction` | 当前帧开始到提交 | T0 私有可变 | 最多两次尝试、计划、临时观测和预算 |
| `StateObservation` | 一次尝试评估 | 不可变 | 聚合结果、分数、证据和结果提案 |
| `StateRecord` | 提交后 | 不可变/可选落盘 | 状态、最终评估、转移、结果和诊断摘要 |
| `ResultSink` | 整个序列 | T3 单写 | 每帧最终 `TrackResult` |

### 9.2 跨状态 `ControllerMemory`

```python
@dataclass(slots=True)
class ControllerMemory:
    sequenceId: SequenceId
    nextFrameIndex: FrameIndex
    stateRevision: int
    nextStateId: int
    nextTransactionId: int

    anchor: AnchorTargetState
    lastConfirmed: ConfirmedTargetState
    motionHistory: deque[MotionSample]
    scaleHistory: deque[ScaleSample]
    depthHistory: deque[DepthSample]

    mode: TrackMode
    modeAgeFrames: int
    stableStreak: int
    weakStreak: int
    missStreak: int
    reacquireCooldownRemaining: int

    recovery: RecoveryMemory
    template: TemplateMemory
    pending: FrameTransaction | None
    committedResultCount: int
```

其中：

- `anchor`：首帧框、BFoV、模板 revision 和可选深度摘要，整个 session 不被在线候选覆盖。
- `lastConfirmed`：最后可靠提交的 frame、bbox、BFoV、中心、分数和深度。
- 三个 history 均有界，只保存数值摘要；可以合并到 `motionHistory`，但不能保留图像。
- `recovery`：跨 `RECOVERING/LOST` 保存搜索进度。
- `template`：只保存槽位元数据和待发命令，不保存模型特征。
- `pending`：唯一未提交帧事务；异常时整体丢弃。

### 9.3 `RecoveryMemory`

```python
@dataclass(slots=True)
class RecoveryMemory:
    epochId: int
    startedFrameIndex: FrameIndex | None
    framesSpent: int
    globalScanPhase: int
    coveredCells: set[SphereCellId]
    attemptedPlanKeys: set[PlanKey]
    bestSeedCenter: SphericalPoint | None
    bestSeedScore: float
    bestSeedFrameIndex: FrameIndex | None
    lastGlobalScanFrameIndex: FrameIndex | None
```

进入新 recovery epoch 时清空覆盖集合；同一 epoch 内跨帧保留，避免当前实现每帧重复相同环。
`PlanKey` 应由量化后的中心、FOV、角色和 epoch 组成，不依赖 Python 对象地址。

### 9.4 `TemplateMemory`

```python
@dataclass(slots=True)
class TemplateMemory:
    backendRevision: int
    anchorAvailable: bool
    recentSourceFrame: FrameIndex | None
    stableSourceFrame: FrameIndex | None
    pendingDecision: TemplateDecision
    lastUpdateFrame: FrameIndex | None
```

模板特征仍由 T2 后端拥有。控制层只保存来源和 revision，用来生成下一请求的 `TemplateCommand`。

### 9.5 状态特有数据

| 状态 | 只在该模式有意义的数据 |
|---|---|
| `INIT` | 初始框、初始化计划、模板响应 |
| `TRACKING` | 稳定 streak、找回冷却、近期可靠尺度 |
| `UNCERTAIN` | weak/miss streak、当前局部扩大倍率、最佳弱候选 |
| `RECOVERING` | recovery epoch、已覆盖球面单元、环层级、已花费帧/视图预算 |
| `LOST` | lost 起始帧、cube-map 相位、上次全局扫描帧、最佳未确认候选 |
| `TERMINATED` | 结束原因、最终帧数、是否已 finalize |

这些数据不应散落成 controller 的大量平行私有字段；优先组合进 `ControllerMemory` 的子结构。

### 9.6 结果和历史如何保存

原文设想状态机最终输出包含每帧框列表的 `TrackingResult`。现有工程已经采用逐帧 `TrackResult + ResultSink`，
V2 应保持该流式协议，避免控制器长期持有整段序列：

- `TrackResult`：一帧一个，提交后立即交给 sink。
- `StateRecord`：可选诊断 JSONL，仅保存小型数值和 ID。
- `TrackingSessionSummary`：序列结束时保存帧数、状态占比、恢复统计和结果文件位置。
- 如果离线调用方确实需要列表，由 app 层 `CollectingResultSink` 组装 `tuple[TrackResult, ...]`，
  不把列表塞回状态机核心。

### 9.7 不保存的内容

- ERP RGB/Depth、局部视图数组、深度伪彩图；
- CUDA tensor、HiT 特征、模板特征；
- 完整 `LocalObservation` 历史；
- 无界候选/计划集合；
- 已发布结果的可变引用。

---

## 10. Controller V2 组件结构

```text
TrackSessionController / DTC (T0, single writer)
  ├─ FrameTransactionCoordinator  帧号、attempt、预算与原子提交
  ├─ SphericalMotionPredictor     窗口历史、预测与不确定度
  ├─ StateAwareViewPlanner        五视图、环搜、cube-map 与去重
  ├─ StateEvaluator               候选评分、聚类、融合、StateObservation
  ├─ TrackStateMachine            纯状态转移 reducer
  ├─ TemplatePolicy               模板命令与找回冷却
  └─ ControllerMemory             唯一跨状态可变数据
```

边界保持不变：

- geometry 只裁剪和回投影；
- TrackerBackend 只进行局部 RGB/RGB-D 跟踪与融合；
- controller 只消费回投影观测与摘要；
- visualization 只读取提交后的诊断副本；
- app 负责 T0/T2 请求循环和 ResultSink。

### 10.1 纯状态机接口

```python
class TrackStateMachine:
    def transition(
        self,
        state: StateInstance,
        observation: StateObservation,
        counters: CounterSnapshot,
        budget: BudgetSnapshot,
    ) -> TransitionDecision: ...
```

`transition()` 不修改自身。输出至少包含：

```python
@dataclass(frozen=True, slots=True)
class TransitionDecision:
    action: ControllerAction  # ESCALATE / COMMIT / TERMINATE
    nextMode: TrackMode
    reason: TransitionReason
    acceptMeasurement: bool
    resetMotionHistory: bool
    resetRecoveryEpoch: bool
    templatePermission: TemplatePermission
```

### 10.2 控制器步骤接口

现有 `plan(frame) -> SearchPlan`、`update(plan, observations) -> TrackResult` 无法表达同帧第二次请求。
建议改为判别联合类型：

```python
ControllerStep = MoreViewsRequired | FrameCommitted

def beginFrame(frame: FramePacket) -> SearchPlan: ...

def consume(
    plan: SearchPlan,
    observations: Sequence[ProjectedObservation],
) -> ControllerStep: ...
```

- `MoreViewsRequired` 携带同一 frame/transaction 的下一 `SearchPlan`。
- `FrameCommitted` 携带唯一 `TrackResult` 和可选 `StateRecord`。
- 如果决定首版不启用同帧升级，可以先实现相同接口但始终返回 `FrameCommitted`，避免以后再次破坏协议。

### 10.3 帧级伪代码

```text
frame = read next frame
plan = controller.beginFrame(frame)

loop:
    views = geometry.cropViews(frame, plan.views)
    local = backend.infer(views, plan.templateCommand)
    projected = geometry project all local observations
    step = controller.consume(plan, projected)

    if step is MoreViewsRequired:
        plan = step.plan
        continue

    sink.write(step.result)
    break
```

循环上限由 controller 事务硬性保证；app 仍应增加防御性断言，超过上限视为内部协议错误。

### 10.4 原子提交顺序

1. 完成响应协议校验。
2. 纯计算得到 `StateObservation`。
3. 纯计算得到 `TransitionDecision`。
4. 若 `ESCALATE`，只更新当前事务尝试记录，不修改跨帧可靠历史。
5. 若 `COMMIT`，先在临时副本中计算新的 motion/recovery/template/counter 状态。
6. 构造并完整校验 `TrackResult`、新 `StateInstance` 和 `StateRecord`。
7. 一次性替换 `ControllerMemory`，revision 增加 1，清空 pending。
8. 发布只读结果。

任何一步失败都保留提交前的 `ControllerMemory`；不得像当前可变状态机一样先改变 mode 再执行可能失败的后续更新。

---

## 11. 输出语义

建议新增内部/诊断枚举：

```python
class ResultSource(Enum):
    INITIAL = auto()
    OBSERVED_CONFIRMED = auto()
    OBSERVED_REACQUIRED = auto()
    OBSERVED_WEAK_BLEND = auto()
    MOTION_PREDICTED = auto()
```

| 来源 | `valid` | 更新运动历史 | 更新模板 |
|---|---:|---:|---:|
| `INITIAL` | true | 初始化 | 初始化 anchor |
| `OBSERVED_CONFIRMED` | true | 是 | 仅稳定 tracking 可更新 |
| `OBSERVED_REACQUIRED` | true | 重建历史 | 冷却期禁止 |
| `OBSERVED_WEAK_BLEND` | false | 否 | 否 |
| `MOTION_PREDICTED` | false | 否 | 否 |

比赛输出仍逐帧给框。`valid=false` 不代表可以少写一行，只说明该框不是可靠观测提交。
若不希望修改公共 `TrackResult`，`ResultSource` 可先放入 `StateRecord`；长期建议作为带默认值的诊断字段加入。

---

## 12. 配置改动蓝图

现有字段应真正接入算法：

- `tracking.uncertainThreshold`
- `tracking.windowLength`
- `tracking.maxPredictionHorizon`
- `tracking.acceptThreshold`
- `tracking.recoverAcceptThreshold`
- `tracking.uncertainPatience`
- `tracking.maxRecoveryFrames`
- `tracking.minViewsForCommit`
- `recovery.maxViewsPerFrame`
- `recovery.globalSearchInterval`

建议新增字段，具体默认值必须通过离线实验确定，本文只给语义：

| 字段 | 语义 |
|---|---|
| `tracking.maxAttemptsPerFrame` | 同帧最大推理尝试，范围 1..2 |
| `tracking.maxViewsPerFrameTotal` | 同帧所有尝试合计视图硬上限 |
| `tracking.sameFrameEscalationEnabled` | 是否允许低分后同帧补充视图 |
| `tracking.reacquireCooldownFrames` | 找回后运动/模板保护期 |
| `tracking.minTrackingViews` | 稳定状态最少视图数 |
| `tracking.uncertainFovScale` | 不确定状态相对 FOV 放大倍率 |
| `motion.minSamplesForVelocity` | 估计速度所需最少可靠样本 |
| `motion.maxTangentSpanRad` | 单个切平面拟合允许的最大跨度 |
| `motion.huberDeltaRad` | 球面残差稳健重加权阈值 |
| `motion.processNoiseRadPerSec` | 预测不确定度增长率 |
| `motion.maxAngularSpeedRadPerSec` | 防异常速度上限 |
| `motion.maxLogScaleRatePerSec` | 尺度变化率上限 |
| `evaluator.supportWeight` | 状态分数的多视图支持权重 |
| `evaluator.agreementWeight` | 状态分数的簇一致性权重 |
| `evaluator.minReacquireViews` | 找回所需的最少独立视图 |
| `recovery.cubemapOverlapRatio` | 六面搜索边界重叠比例 |
| `recovery.maxCoveredCells` | 恢复去重集合的内存上限 |
| `runtime.frameInferenceBudgetMs` | 可选的帧级推理软时限；比赛确定性模式可关闭 |

所有新增字段必须同步严格 schema、两份 YAML、`docs/hyperparameters.md` 和配置单测；
不得在算法中加入未登记的隐藏常量。

---

## 13. V2 实施落点

以下核心修改已经落地。可选的状态诊断图片仍保持默认关闭，不影响 V2 闭环。

### 13.1 核心与协议

| 文件 | 已实施修改 |
|---|---|
| `src/instatarget/core/types.py` | 增加必要的 attempt/transaction/result source 只读类型；避免把 controller 私有细节全部放入 core |
| `src/instatarget/core/protocols.py` | `TrackController` 改为可返回 `MoreViewsRequired | FrameCommitted` |
| `src/instatarget/core/config.py` | 接入第 12 节配置和严格范围/组合校验 |
| `configs/RGBonly.yaml` | 增加相同控制结构，深度证据自动缺省 |
| `configs/RGBD.yaml` | 增加相同控制结构并启用深度一致性 |

### 13.2 Controller

| 文件 | 已实施修改 |
|---|---|
| `controller/state_machine.py` | 重写为无内部可变状态的纯 reducer；补完整转移原因 |
| `controller/state_evaluator.py` | 新增；实现协议校验、候选证据、无顺序依赖聚类、稳健融合和 `StateObservation` |
| `controller/state_model.py` | 建议新增；集中定义 controller 私有状态实例、事务、内存和记录类型 |
| `controller/motion_estimator.py` | 改为显式滑动窗口、球面切平面拟合、尺度/range 和不确定度；找回可重置 |
| `controller/recovery_planner.py` | 状态相关五视图、真正六面 cube-map、跨帧 coveredCells、attempt 去重和总预算 |
| `controller/decision_gate.py` | 将聚合职责迁入 evaluator；可保留无状态候选打分原语，避免双重门控 |
| `controller/template_policy.py` | 增加找回冷却、anchor 复核和 `TemplateMemory` |
| `controller/depth_aware_track_controller.py` | 改为事务协调器和单一 `ControllerMemory`，实现 copy-on-write 原子提交 |
| `controller/__init__.py` | 导出稳定公共 façade，内部状态类型默认不对外暴露 |

### 13.3 App、消息和可视化

| 文件 | 已实施修改 |
|---|---|
| `src/instatarget/app/driver.py` | 支持同帧有界的 `MoreViewsRequired` 循环；最终只写一次结果 |
| `src/instatarget/core/types.py` 中线程消息 | 请求/响应增加 transactionId、attemptIndex 和 recoveryEpochId |
| `src/instatarget/visualization/*` | 可选记录 prediction、attempt、candidate cluster、stateScore 和 transition reason |
| `src/instatarget/io/result_sink.py` | 比赛输出保持不变；可选增加独立状态诊断 sink，不混入结果文本 |

### 13.4 已同步的现有文档

| 文档 | 已同步内容 |
|---|---|
| `docs/interface.md` | 新 controller step、transaction/attempt、结果来源和严格响应匹配 |
| `docs/modules/controller.md` | 以本文的组件、状态、评估器、预测器和恢复内存替换当前简化描述 |
| `docs/process.md` | T0/T2 同帧最多两轮的有界时序、总预算和停止语义 |
| `docs/design.md` | 状态相关覆盖替代“所有状态固定 guard triplet”的硬编码描述 |
| `docs/hyperparameters.md` | 登记第 12 节字段及实验记录 |
| `docs/modules/visualization.md` | 增加 V2 状态诊断项，保持旁路只读 |
| `docs/implement.md` | 重构完成后再更新实际交付状态，未完成前不得宣称落地 |

`docs/Prepare/StateMachine.md` 保留为原始提案；本文是已实现结构的规范和后续演进依据。

---

## 14. 实施顺序

1. 先增加 controller 私有状态模型和纯 `StateEvaluator` 单测，不改运行入口。
2. 重写运动预测器，验证窗口长度确实影响结果，并覆盖经线、极点、缺深度、离群和找回重置。
3. 重写纯状态机 reducer，用转移表参数化测试覆盖所有状态×证据组合和阈值边界。
4. 扩展 view planner：五视图、环搜去重、六面 cube-map、视图角色和总预算。
5. 将 `DepthAwareTrackController` 改成 copy-on-write `ControllerMemory + FrameTransaction`。
6. 更新协议和 driver，使其支持 `MoreViewsRequired`，先在单线程链路跑通。
7. 接回模板策略和可选可视化，验证诊断不影响结果。
8. 再更新四线程消息，实现 transaction/attempt/revision 全校验。
9. 最后同步现有文档、YAML、配置索引和交付状态。

建议分两阶段落地：

- V2-A：先禁止同帧升级（`maxAttemptsPerFrame=1`），完成数据模型、StateEvaluator、真实多帧预测和纯状态机。
- V2-B：再启用一次同帧升级，改 driver/thread 消息并做性能预算实验。

这样可以先获得结构正确性，再单独评估即时找回是否值得其最坏时延成本。

---

## 15. 测试蓝图

### 15.1 状态机表驱动测试

- 每个活动状态对四级证据的下一状态、结果有效性和模板许可。
- 分数恰好等于 `uncertainThreshold/acceptThreshold/recoverAcceptThreshold`。
- `uncertainPatience/maxRecoveryFrames` 前一帧、等于和后一帧。
- `TRACKING -> TRACKING` 产生新 stateId。
- 找回后 cooldown 和 history reset。
- EOF 只终止，不多输出一帧。

### 15.2 帧事务测试

- 第一次低分、第二次成功：只发布第二次确定的一个结果。
- 两次都失败：必须提交预测并结束事务。
- `maxAttemptsPerFrame=1` 时从不请求第二批。
- 两次视图合计不超总预算，viewId 不重复。
- 旧 transaction、旧 attempt、重复响应、第二次提交全部拒绝。
- 第二次推理异常时跨帧可靠内存保持提交前状态。

### 15.3 StateEvaluator 测试

- 跨经线相同目标形成一个簇。
- 高分单图离群不压过多视图一致簇。
- 调换候选输入顺序不改变结果。
- 相交候选融合框不等于无界最大并集。
- 无相交时最高分只成为弱种子，支持不足不能确认。
- 深度缺失时权重重新归一化；异常深度不污染 range。
- candidate 贴边/裁切、尺度突变和锚点冲突能给出明确 rejection reason。

### 15.4 运动预测测试

- `windowLength` 改变实际消费样本数。
- yaw 从 `+π` 跨到 `-π` 连续预测。
- 近南北极运动不产生 yaw 数值爆炸。
- 匀速、转向、不同帧间隔和单个离群点。
- 无深度启动后首次有效深度不会产生虚假高速 rangeVelocity。
- 连续缺失时置信度下降、不确定度扩大。
- `valid=false` 输出不进入历史。
- 找回后旧速度和未来假设被清除。

### 15.5 视图与恢复测试

- tracking 默认中心 + 四角共五图且覆盖重叠正确。
- cube-map 包含四赤道面和南北极面，并覆盖经线/极点边界。
- `coveredCells` 跨 recovery 帧生效，新 epoch 正确清空。
- 输出分辨率固定，FOV 随尺度/不确定度变化并受 min/max 限制。
- RGB-only/RGB-D 的视图几何完全一致。

### 15.6 集成和性能测试

- 每个输入帧恰好一个输出，帧号连续。
- 单线程与四线程逐帧结果一致。
- 同一批和确定性分批结果一致。
- 同帧升级开启/关闭分别记录 AUC、SR@0.5、FPS、P95 帧时延、恢复时延和误找回率。
- 六面搜索覆盖极区、经线、遮挡、快速转向和长时消失回归样例。

---

## 16. 验收标准

- 状态模式、状态实例、帧内尝试和输出四层语义明确，不再用 `LOST` 表示一次同帧重试。
- `EverLost` 被有界事务完整替代，任何输入都不存在同帧无限循环。
- `StateEvaluator` 能产出本文定义的 `StateObservation`，并能解释每次状态转移原因。
- `windowLength`、`uncertainThreshold`、`maxPredictionHorizon` 均有真实算法行为和测试证明。
- 运动预测只消费可靠测量中心，不消费预测输出；找回后重建窗口。
- recovery 跨帧记住已搜索区域，全局搜索包含南北极面。
- 每帧只有一次原子提交；旧 revision/attempt/transaction 不会改变状态。
- 状态、历史和日志不长期保存图像或模型 tensor。
- RGB-only 与 RGB-D 使用同一控制逻辑，深度缺失自然退化。
- 模板在不确定、恢复、丢失和找回冷却期不会更新。
- 所有后续需要修改的代码、配置、测试和文档均已在本文声明。

---

## 17. 原 `StateMachine.md` 重要注释映射

| 原文要点 | V2 落点 |
|---|---|
| 一个目标任务唯一 state machine | `TrackSessionController + ControllerMemory` 每 session 唯一 |
| 每次进入同名状态也是不同 state | 每帧新建不可变 `StateInstance/stateId` |
| 每个 state 有自己的 ViewSpec | 状态实例关联唯一事务；每个 attempt 有自己的不可变 `SearchPlan` |
| ViewSpec 与 LocalObservation 一一对应 | 正常响应严格数量、顺序、viewId 一一对应 |
| LocalObservation list 经 StateEvaluator 变 StateObservation | 第 7 节完整定义 |
| StateObservation 决定状态切换 | 纯 `TrackStateMachine.transition()` 消费它 |
| 每个 state 有 PredictedCenter | `StateInstance.prediction.center`；INIT 除外 |
| ResultCenter 不同于 PredictedCenter | 拆成 measured/output/search seed/last confirmed 五种中心 |
| 多帧模块使用 ResultCenter 而非 PredictedCenter | 只使用可靠 `measuredCenter`，进一步防止自反馈漂移 |
| tracking/uncertain 五图 | 主图 + 四个切平面角保护图，FOV 自适应 |
| lost/recovery 六图 cube-map | 真正四赤道面 + 南北极面，带跨帧扫描相位 |
| Lost 返回后覆盖本帧 TrackingResult | 帧内最多一次 escalation，最终只发布一个结果，不覆盖已发布结果 |
| EverLost 防止同帧卡死 | `attemptIndex/maxAttemptsPerFrame/预算` 从结构上保证终止 |
| Lost 的 ResultCenter 作为返回搜索中心 | 保存为 `RecoveryMemory.bestSeedCenter`，只用于搜索，不直接更新运动历史 |
| 相交取并集，不相交取最高分 | 并集只做诊断包络；稳健球面融合；单图最高分仅为弱种子 |
| 每张图尽量使用最大视场 | 保留完整输出分辨率；FOV 按状态自适应，cube-map 才使用接近最大视场 |

本文至此形成后续状态机和 controller 重构的完整蓝图。实现时若需要改变本文的不变量、
状态语义或数据所有权，必须先更新本文并记录原因，再修改代码。
