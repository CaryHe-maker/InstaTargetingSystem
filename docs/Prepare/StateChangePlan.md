# Controller 状态机与分状态查询改造计划

## 1. 文档用途

本文是下一轮实现 controller 重构的执行规范。适用范围包括：

- `DepthAwareTrackController`
- `RecoveryPlanner`
- `StateEvaluator`
- `TrackStateMachine`
- controller 私有状态模型、配置、driver、测试和关联文档

实现不得继续沿用当前的自适应 FOV、所有状态共享同一搜索流程、五视图、恢复环搜索或同帧最多
两轮的旧逻辑。若代码实现需要改变本文中的视图数、阈值、终止条件、输出规则或状态转移，必须先
修改本文并记录原因。

## 2. 两层状态必须分离

controller 必须明确区分两类状态：

1. 跨帧状态 `TrackMode`
   包括 `TRACKING`、`UNCERTAIN`、`RECOVERING`、`LOST`，负责描述目标可靠性、运动历史、模板权限和
   连续帧计数。
2. 帧内查询轮次 `attemptIndex`
   包括 `ROUND_1/ROUND_2/ROUND_3`，只负责本帧应该请求哪些 ViewSpec 以及何时结束本帧。

帧内轮次不是 `TrackMode`。前一轮失败只创建下一轮 `SearchPlan`，不得立即修改跨帧状态、运动历史、
模板、弱帧计数或恢复计数。每个输入帧只在最终候选确定后调用一次
`TrackStateMachine.transition()`，并且只提交一个 `TrackResult`。

一帧采用哪条查询路线，由 `beginFrame()` 时的 `TrackMode` 固定。该帧尚未提交前，不能因为某一轮
结果临时切换路线。

## 3. 全局不变量和超参数

### 3.1 固定 120° 最大视域

所有局部 `ViewSpec` 的水平和垂直 FOV 都固定为：

```text
horizontalFov = 120° = 2π/3
verticalFov   = 120° = 2π/3
```

该规则适用于：

- `InitializationPlan.templateView`
- 各状态按照第 3.3 节实际存在的第一轮、第二轮或第三轮搜索视图
- 四角局部搜索视图
- 前、后、左、右、上、下六面 cubemap 视图

禁止根据目标框大小、运动不确定度、状态、轮次或恢复时间改变 FOV。输出像素尺寸仍固定使用
`geometry.viewWidthPx × geometry.viewHeightPx`，也不进行逐轮缩放。

`geometry.maxFovDeg` 必须配置为 `120.0`。下一轮实现应删除或停止使用仅服务于旧搜索 FOV 放大的
`_contextFov()`、`uncertainFovScale`、恢复环倍率和 cubemap FOV overlap 计算。

### 3.2 阈值定义

新增 evaluator 超参数：

```yaml
evaluator:
  successRate: 0.90
  firstRoundFusionOverlap: 0.30
  overlapThreshold: 0.70
  fusionSourceMinConfidence: 0.80
```

本文将 `evaluator.overlapThreshold` 称为 `OverlapThreshold`，默认值为 `0.70`。所有原本承担
“高重合融合”“融合框可输出”或“可靠融合证据”含义的固定阈值，统一改用
`OverlapThreshold`，不得在代码中再次写死相同数值。

除 `FusionSourceMinConfidence` 使用大于等于号外，以下 overlap 和 success 阈值全部采用严格大于号：

- 普通第一轮允许在 `OverlapRate > firstRoundFusionOverlap` 时生成 FuseBox。
- 后续轮次以及 LOST 第一轮只在 `OverlapRate > OverlapThreshold` 时生成 FuseBox。
- 融合框只有在 `OverlapRate > OverlapThreshold` 时才属于可靠融合证据。
- 成功条件使用 `candidate.confidence > SuccessRate`。
- 对上述 overlap 和 success 条件，刚好等于阈值不算通过。

`SuccessRate`、`OverlapThreshold` 和 `FusionSourceMinConfidence` 必须只从配置注入
`StateEvaluator`，不能由 planner、状态机或 driver 分别维护副本。

`evaluator.fusionSourceMinConfidence` 称为 `FusionSourceMinConfidence`，默认值为 `0.80`。它不是
新的置信度计算公式，而是 FuseBox 的源框可靠性门控：

```text
sourceConfidencePassed = min(a, b) >= FusionSourceMinConfidence
```

这里按“0.8 以上”使用大于等于号。该门控不影响 FuseBox 的生成，也不改变融合置信度；它只决定
FuseBox 能否成为 `RELIABLE_FUSED`、在第一轮提前输出、完成找回或更新可靠历史。未通过门控的
FuseBox 仍保留在候选池，最终轮仍可按最高分输出，但其证据只能分类为 `WEAK`。

### 3.3 每种状态的轮次和视图预算

| 帧开始状态 | 第一轮 | 第二轮 | 第三轮 | 最大视图数 |
|---|---|---|---|---:|
| `TRACKING` | 4 张四角局部视图 | 4 张四角局部视图，最终轮 | 无 | 8 |
| `UNCERTAIN` | 4 张四角局部视图 | 4 张四角局部视图 | 6 张 cubemap，最终轮 | 14 |
| `RECOVERING` | 4 张四角局部视图 | 4 张四角局部视图 | 6 张 cubemap，最终轮 | 14 |
| `LOST` | 6 张 cubemap | 4 张四角局部视图，最终轮 | 无 | 10 |

只有 `UNCERTAIN` 和 `RECOVERING` 可以进入第三轮。`TRACKING` 和 `LOST` 的第二轮无条件结束本帧，
任何分支都不得为它们创建第三轮计划。

配置层仍保留全局硬上限：

```yaml
tracking:
  maxAttemptsPerFrame: 3
  maxViewsPerFrameTotal: 14
```

该上限只表示系统最大能力。planner/controller 仍必须按照上表在 `TRACKING`、`LOST` 中提前终止。

## 4. ViewSpec 生成规则

### 4.1 多帧预测中心 c1

除固定 cubemap 外，每帧先由多帧运动预测模块生成：

```text
c1 = MotionPrediction.center
```

`TRACKING`、`UNCERTAIN`、`RECOVERING` 第一轮以 `c1` 为四角布局中心。`LOST` 第一轮直接执行固定
cubemap，不围绕 `c1` 排布；当 LOST 第一轮没有任何候选时，`c1` 作为第二轮中心的防御性回退。

### 4.2 四角局部视图

第一或第二轮需要四角搜索时，以本轮 seed 为中心建立局部 right/up/forward 正交基。每个视域均为
`120° × 120°`。

相邻视域沿相邻轴重合单视域角覆盖的 `1/3`：

```text
adjacentCenterDistance = 120° × (1 - 1/3) = 80°
halfOffset = 80° / 2 = 40°
```

四个视域中心位于局部角坐标正方形四角：

```text
left_top     = (-40°, +40°)
right_top    = (+40°, +40°)
left_bottom  = (-40°, -40°)
right_bottom = (+40°, -40°)
```

因此相邻视域单轴共同覆盖为 `40°/120° = 1/3`，四图中心共同覆盖为
`(1/3) × (1/3) = 1/9`，seed 是该共同区域中心。不额外生成中心视图。

偏移必须通过局部球面旋转生成，不能直接执行全局 `yaw += offset`、`pitch += offset`，否则经线和
极点附近会失去正方形对称性。建议增加：

```python
offsetDirection(center, localYawOffsetRad, localPitchOffsetRad) -> SphericalPoint
```

第一轮和第二轮 role 应分别明确为 `round1_left_top/...` 与 `round2_left_top/...`。

### 4.3 六面 cubemap

cubemap 固定覆盖全景坐标系：

```text
front: yaw=0,     pitch=0
right: yaw=+π/2, pitch=0
back:  yaw=-π,   pitch=0
left:  yaw=-π/2, pitch=0
up:    yaw=0,     pitch=+π/2
down:  yaw=0,     pitch=-π/2
```

每个面使用 `120° × 120°`。禁止根据 c1、候选中心、recovery epoch 或 global scan phase 旋转六面
布局。`LOST` 第一轮与 `UNCERTAIN/RECOVERING` 第三轮复用同一个 cubemap 构造函数，仅 role 和
attemptIndex 不同。

## 5. 候选框与双框融合

### 5.1 原始候选置信度

每个局部预测回投影为一个 `ProjectedObservation`。候选排序和融合公式中的源框置信度直接使用：

```text
candidateConfidence = ProjectedObservation.fusedScore
```

现有 `DecisionGate.decisionScore`、motion/scale/depth 分数可以保留为诊断信息或最终跨帧可靠性分析，
但不能改变 FuseBox 公式、候选最高分排序或 `SuccessRate` 比较。原 `candidateMinScore` 不得在融合前
删除候选。

### 5.2 预测框 OverlapRate

两个不同 `viewId` 的回投影预测框 A、B 使用：

```text
OverlapRate(A, B) = intersectionArea(A, B) / min(area(A), area(B))
```

这不是 IoU，也不是第 4.2 节的 ViewSpec 覆盖比例。计算必须处理 ERP 左右经线跨越。优先在球面
BFoV 上计算；如果使用 ERP bbox，必须先拆分跨经线区间，不能把跨经线同一目标判为不相交。结果
限制在 `[0, 1]`。

### 5.3 一对一配对

融合最多包含两个局部框，且一个局部框最多参与一个 FuseBox：

1. 枚举同一轮、不同 `viewId` 的全部候选对。
2. 删除未严格超过本轮融合阈值的候选对。
3. 按 `OverlapRate` 降序排序；同值时依次按较高源置信度、较小 viewId 排序。
4. 依次选择两端都未参与其他 FuseBox 的候选对。
5. 任一端已被使用则跳过，不寻找三框融合，也不让 FuseBox 再次参与融合。

该全局降序贪心规则用于解决多个框争用同一伙伴的冲突，并保证输入顺序不影响配对结果。

### 5.4 FuseBox

FuseBox 是完整包含两个源预测框的 seam-aware 最小包围框，不是交集，也不是当前实现的加权中位
BFoV。它必须显式记录：

```python
@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    bfov: BFoV
    bbox: BBoxXYWH
    confidence: float
    sourceViewIds: tuple[int, ...]       # 局部框 1 个；FuseBox 恰好 2 个
    fused: bool
    overlapRate: float | None
    minSourceConfidence: float | None
    sourceConfidencePassed: bool
    representativeViewId: int
    representativeLocalBox: BBoxXYWH | None
    depthSummary: DepthSummary | None
```

源框置信度为 a、b，OverlapRate 为 y 时：

```python
fusedConfidence = 1.0 - ((2.0 - b - a) * (1.0 - y) / 2.0)
fusedConfidence = clip(fusedConfidence, 0.0, 1.0)
```

FuseBox 必须保存 `fused=True`、两个 `sourceViewIds` 和实际 `overlapRate`。两个源局部框生成 FuseBox
后仍保留在本轮候选池，与 FuseBox 一起竞争最高置信度；“最多融合两张”只限制 FuseBox 的来源，
不删除原始局部框。

FuseBox 还必须保存 `minSourceConfidence = min(a, b)` 和
`sourceConfidencePassed = (minSourceConfidence >= FusionSourceMinConfidence)`。这两个字段用于
最终可靠性分类，不能回写或替换 `fusedConfidence`。

`representativeViewId/localBox` 选择两个源框中置信度较高者，同分选择较小 viewId。它仅服务模板
和诊断，不能把球面 FuseBox 错当成单张局部视图里的 bbox。

## 6. 各状态的帧内查询流程

### 6.1 通用候选处理顺序

每轮严格按以下顺序处理：

1. 校验 frame、transaction、attempt、stateRevision、viewId 和响应顺序。
2. 读取所有原始局部候选及其 `fusedScore`。
3. 按本轮阈值生成一对一 FuseBox。
4. 候选池设为“所有原始局部框 + 所有 FuseBox”。
5. 按 `(confidence, fused, -representativeViewId)` 确定性排序；置信度优先，同分时 FuseBox 优先，
   再选择较小 viewId。
6. 得到本轮唯一 `bestCandidate`，再根据状态和轮次判断输出或继续。

不得跳过全局最高候选，转而选择排名更低但更容易满足输出条件的候选。

### 6.2 TRACKING：最多两轮

第一轮：

- 以 `c1` 为中心生成 4 张四角视图。
- 使用 `firstRoundFusionOverlap` 作为生成 FuseBox 的门槛。
- 取全局最高候选 `best1`。
- 仅当 `best1` 同时满足以下条件时立即输出：
  - `best1.fused is True`
  - `best1.overlapRate > OverlapThreshold`
  - `best1.confidence > SuccessRate`
  - `best1.sourceConfidencePassed is True`
- 否则以 `best1.bfov.center` 作为 `c2` 进入第二轮。
- 第一轮局部框即使高于 `SuccessRate` 也不能直接输出。

第二轮：

- 以 `c2` 为中心生成 4 张相同布局的最大视域。
- 只融合 `OverlapRate > OverlapThreshold` 的候选对。
- 直接输出候选池中置信度最高的 `best2`，不再比较 `SuccessRate`。
- 第二轮是 TRACKING 的最终轮，不允许进入 cubemap 第三轮。

如果第二轮结果较弱，先提交本帧输出，再由跨帧状态机把下一帧状态改成 `UNCERTAIN`；只有下一帧
以 `UNCERTAIN` 开始后，才允许使用第三轮。

### 6.3 UNCERTAIN：最多三轮

第一轮与 TRACKING 第一轮完全相同。未满足第一轮成功条件时，以 `best1` 中心进入第二轮。

第二轮：

- 以 `c2` 为中心生成 4 张四角视图。
- 只融合 `OverlapRate > OverlapThreshold` 的候选对。
- 如果最高候选 `best2.confidence > SuccessRate`，立即输出；局部框和 FuseBox 都可以输出。
- 否则进入第三轮。

第三轮：

- 请求固定 6 张 `120°` cubemap。
- 只融合 `OverlapRate > OverlapThreshold` 的候选对。
- 直接输出置信度最高的 `best3`，不比较 `SuccessRate`。
- 第三轮无条件终止，不存在第四轮。

### 6.4 RECOVERING：最多三轮

RECOVERING 的 ViewSpec、融合和轮次停止规则与 UNCERTAIN 完全一致：四角第一轮、四角第二轮、必要
时六面第三轮。

区别只在最终候选提交后的跨帧状态转移：RECOVERING 必须执行找回确认、运动历史重置和模板冷却，
不能因为搜索流程相同就复用 UNCERTAIN 的状态更新结果。

### 6.5 LOST：六面第一轮，局部第二轮

第一轮：

- 直接请求固定前、后、左、右、上、下 6 张 `120°` cubemap。
- 不围绕 c1 生成四角视图。
- 只融合 `OverlapRate > OverlapThreshold` 的候选对。
- 取全局最高候选 `best1`。
- 若 `best1` 是可靠融合候选，即 fused、overlap 超过 `OverlapThreshold`、confidence 超过
  `SuccessRate` 且 `sourceConfidencePassed=True`，可以立即输出并交给状态机判断为可靠找回。
- 否则以 `best1.bfov.center` 作为 `c2` 进入第二轮。

第二轮：

- 以 `c2` 为中心生成与其他状态相同的 4 张四角最大视域。
- 只融合 `OverlapRate > OverlapThreshold` 的候选对。
- 直接输出候选池中置信度最高的 `best2`，不比较 `SuccessRate`。
- LOST 没有第三轮，第二轮后必须结束本帧。

### 6.6 空候选防御性规则

需求中的“最高候选”要求候选池非空，但 HiT 或投影链路可能返回空集合。为保证事务严格有界：

- 任一第一轮无候选：第二轮中心回退为 `c1`。
- UNCERTAIN/RECOVERING 第二轮无候选：继续进入 cubemap 第三轮。
- TRACKING/LOST 第二轮无候选：输出运动预测框，`valid=False`，结束本帧。
- UNCERTAIN/RECOVERING 第三轮无候选：输出运动预测框，`valid=False`，结束本帧。

存在候选时不得用运动预测框替换最终轮最高候选。

## 7. 最终证据分类

帧内最终候选确定后，`StateEvaluator` 将其分类为四级证据。证据只描述最终结果，不用于回头改变
本帧已经执行的查询路线。

```python
class MeasurementEvidence(Enum):
    RELIABLE_FUSED = auto()
    RELIABLE_SINGLE = auto()
    WEAK = auto()
    MISSING = auto()
```

分类规则：

- `RELIABLE_FUSED`：最终候选是 FuseBox，`overlapRate > OverlapThreshold`、
  `confidence > SuccessRate` 且 `sourceConfidencePassed=True`。
- `RELIABLE_SINGLE`：最终候选是局部框且 `confidence > SuccessRate`。
- `WEAK`：存在最终候选，但不满足上述可靠条件；包括融合总分很高但任一源框低于
  `FusionSourceMinConfidence` 的 FuseBox。
- `MISSING`：最终轮没有候选，只能输出 motion fallback。

第一轮能提前成功的候选按定义只能是 `RELIABLE_FUSED`。最终轮“必须输出”不等于证据可靠：低分
候选仍输出框，但分类为 `WEAK`。

## 8. 跨帧状态转移

### 8.1 状态含义

- `TRACKING`：最近一次提交是可靠测量，允许正常维护运动历史。
- `UNCERTAIN`：仍有输出，但证据暂时不足；冻结模板，等待最多若干帧重新确认。
- `LOST`：没有可靠测量；采用六面优先搜索，弱输出不进入运动历史。
- `RECOVERING`：LOST 后出现可靠单框，但尚未得到多视图融合或连续帧确认；冻结模板和旧运动趋势。

`REACQUIRED` 应作为一次 transition reason/event，而不是长期状态。

### 8.2 状态转移表

| 当前状态 | `RELIABLE_FUSED` | `RELIABLE_SINGLE` | `WEAK` | `MISSING` |
|---|---|---|---|---|
| `TRACKING` | `TRACKING` | `TRACKING` | `UNCERTAIN` | `UNCERTAIN` |
| `UNCERTAIN` | `TRACKING` | `TRACKING` | 未超 patience 时保持 `UNCERTAIN`，否则 `LOST` | `LOST` |
| `LOST` | `TRACKING`，直接找回 | `RECOVERING`，开始确认 | `LOST` | `LOST` |
| `RECOVERING` | `TRACKING`，确认找回 | 连续确认达到要求后 `TRACKING`，否则保持 `RECOVERING` | `LOST` | `LOST` |

新增超参数：

```yaml
tracking:
  uncertainPatience: 2
  recoverConfirmFrames: 2
```

严格计数语义：

- TRACKING 首次得到 `WEAK/MISSING` 后提交该帧，并令下一帧从 UNCERTAIN 开始；同时将
  `weakStreak` 设为 1。
- UNCERTAIN 的 `WEAK` 连续次数小于 `uncertainPatience` 时保持 UNCERTAIN；达到该值时进入 LOST。
- UNCERTAIN 的 `MISSING` 直接进入 LOST，不再等待 patience。
- LOST 的 `RELIABLE_SINGLE` 进入 RECOVERING，并将连续单框确认计数设为 1。
- RECOVERING 再次得到 `RELIABLE_SINGLE`，计数达到 `recoverConfirmFrames` 后进入 TRACKING。
- 任意 `RELIABLE_FUSED` 都可完成可靠确认；从 LOST/RECOVERING 返回 TRACKING 时 transition reason
  为 `REACQUIRED`。
- RECOVERING 一旦得到 `WEAK/MISSING`，确认链中断并返回 LOST。

### 8.3 输出、valid、运动历史和模板

状态转移和输出框是两个不同决策。最终轮总会尽量输出框，但可靠性规则如下：

| 最终情况 | 发布 bbox | `valid` | 更新运动历史 | 更新模板 |
|---|---:|---:|---:|---:|
| TRACKING/UNCERTAIN 中可靠证据并转入 TRACKING | 是 | true | 是 | 仅满足稳定帧策略后 |
| LOST 的可靠融合并直接找回 | 是 | true | 重置后写入 | 否，进入 cooldown |
| LOST/RECOVERING 的可靠单框但仍在确认 | 是 | false | 否 | 否 |
| RECOVERING 确认完成 | 是 | true | 清除旧趋势后重建 | 否，进入 cooldown |
| `WEAK` | 是 | false | 否 | 否 |
| `MISSING` motion fallback | 是 | false | 否 | 否 |

弱候选可作为当前结果和下一帧搜索诊断，但不能成为多帧预测模块的新测量样本。模板在 UNCERTAIN、
LOST、RECOVERING 和找回 cooldown 中必须冻结。

## 9. StateObservation 和事务模型

### 9.1 StateObservation

现有 `StateObservation` 应增加或调整以下显式字段：

```python
@dataclass(frozen=True, slots=True)
class StateObservation:
    # existing identity fields ...
    attemptIndex: int
    evaluatedMode: TrackMode
    isFinalAttempt: bool
    successRate: float
    fusionThreshold: float
    overlapThreshold: float
    fusionSourceMinConfidence: float
    bestCandidate: EvaluatedCandidate | None
    candidateCount: int
    fusedCandidateCount: int
    selectedIsFused: bool
    selectedOverlapRate: float | None
    selectedMinSourceConfidence: float | None
    selectedSourceConfidencePassed: bool
    evidence: MeasurementEvidence | None
    outputEligible: bool
    escalationRecommended: bool
```

语义要求：

- 前几轮 `bestCandidate` 用于决定成功或下一轮中心，但 `evidence` 可以暂不分类，不能提交跨帧状态。
- `measuredBfov/measuredBbox/measuredCenter` 指向本轮最高候选，即使该轮没有成功。
- `stateScore` 直接等于最高候选 confidence，不再乘 support/agreement 权重。
- `selectedIsFused/selectedOverlapRate` 必须显式保存，不能只从 sourceViewIds 猜测。
- FuseBox 的最小源框置信度和门控结果必须进入 observation，保证状态转移、诊断和测试可以确认
  `FusionSourceMinConfidence` 确实生效。
- `escalationRecommended` 由“起始 TrackMode + attemptIndex + 本轮成功条件”决定。
- TRACKING/LOST 第二轮以及 UNCERTAIN/RECOVERING 第三轮的
  `isFinalAttempt=True`、`escalationRecommended=False`。
- `evidence` 只在最终提交 observation 中非空，并按第 7 节计算。

### 9.2 FrameTransaction

`FrameTransaction` 必须保存 `startingMode`，所有轮次路线都以它为准。当前
`escalationUsed: bool` 只能表达一次升级，应删除或替换为 `attemptIndex/completedAttempts`。

每个 attempt 的 viewId 在同一事务内连续且唯一：

- TRACKING：`0..3`、`4..7`
- UNCERTAIN/RECOVERING：`0..3`、`4..7`、`8..13`
- LOST：`0..5`、`6..9`

若事务使用非零 viewId 起点则整体平移，但不得重复。前几轮 `AttemptRecord` 只用于有界诊断，不得
长期保存图像或 tensor。

## 10. Controller 伪代码

```text
beginFrame(frame):
    prediction = motion.predictDetailed(frame)
    tx = FrameTransaction(startingMode=currentMode, prediction=prediction)

    if currentMode == LOST:
        return cubemapPlan(attempt=0, fov=120°)
    return fourCornerPlan(center=prediction.center, attempt=0, fov=120°)

consume(round1, observations):
    threshold = OverlapThreshold if tx.startingMode == LOST
                else firstRoundFusionOverlap
    obs1 = evaluator.evaluate(threshold)

    if obs1.bestCandidate is reliable fused:
        return finalizeFrame(obs1)

    c2 = obs1.bestCandidate.center if present else tx.prediction.center
    return MoreViewsRequired(fourCornerPlan(c2, attempt=1, fov=120°))

consume(round2, observations):
    obs2 = evaluator.evaluate(OverlapThreshold)

    if tx.startingMode in {TRACKING, LOST}:
        return finalizeFrame(obs2)              # no confidence gate, no round 3

    if obs2.bestCandidate.confidence > SuccessRate:
        return finalizeFrame(obs2)

    return MoreViewsRequired(cubemapPlan(attempt=2, fov=120°))

consume(round3, observations):
    assert tx.startingMode in {UNCERTAIN, RECOVERING}
    obs3 = evaluator.evaluate(OverlapThreshold)
    return finalizeFrame(obs3)                  # no confidence gate, no round 4

finalizeFrame(finalObservation):
    evidence = evaluator.classifyFinal(finalObservation)
    decision = stateMachine.transition(tx.startingMode, evidence, counters)
    result = atomicallyCommit(decision, finalObservation)
    return FrameCommitted(result)
```

`TrackStateMachine.transition()` 每帧只能在 `finalizeFrame()` 中调用一次。

## 11. 配置迁移

下一轮至少同步以下配置：

```yaml
geometry:
  maxFovDeg: 120.0

evaluator:
  successRate: 0.90
  firstRoundFusionOverlap: 0.30
  overlapThreshold: 0.70
  fusionSourceMinConfidence: 0.80

tracking:
  uncertainPatience: 2
  recoverConfirmFrames: 2
  maxAttemptsPerFrame: 3
  maxViewsPerFrameTotal: 14
```

必须同时修改 `core/config.py` 的严格 schema、`configs/RGBonly.yaml`、`configs/RGBD.yaml`、
`docs/hyperparameters.md` 和配置单测。

下列旧字段不再决定搜索 ViewSpec，应删除或明确废弃：

- `tracking.uncertainFovScale`
- `tracking.guardYawStepDeg`（若无其他使用者）
- `recovery.ringRadii`
- `recovery.viewsPerRing`
- `recovery.cubeMapOverlapRatio`
- `recovery.globalSearchInterval`
- 仅用于缩放搜索 FOV 的 `tracking.contextScale/contextMarginRatio`

原 `acceptThreshold/uncertainThreshold/recoverAcceptThreshold` 若仍供兼容 API 使用，不得参与新的帧内
成功条件；新路径统一使用 `SuccessRate` 和最终四级证据。

## 12. 下一轮逐文件实施顺序

1. `src/instatarget/core/config.py`
   新增 `successRate`、`firstRoundFusionOverlap`、`overlapThreshold`、`fusionSourceMinConfidence`、
   `recoverConfirmFrames`，更新三轮和 14 视图硬上限校验，清理废弃字段。
2. `configs/RGBonly.yaml`、`configs/RGBD.yaml`、`docs/hyperparameters.md`
   同步严格 schema 和默认值。
3. `src/instatarget/controller/state_model.py`
   增加 `EvaluatedCandidate`、`MeasurementEvidence`、新的 `StateObservation` 字段、startingMode 和恢复
   确认计数。
4. `src/instatarget/controller/recovery_planner.py`
   用固定 `fourCornerPlan()` 与 `cubemapPlan()` 替换五视图、状态 FOV 放大、环搜、phase 和旧全局扫描。
5. `src/instatarget/geometry/*`
   增加局部球面偏移、seam-aware OverlapRate 和两个 BFoV 的最小包络函数。
6. `src/instatarget/controller/decision_gate.py`
   保留需要的单框诊断打分；停止使用 connected cluster 和 weighted-median aggregate 作为候选选择器。
7. `src/instatarget/controller/state_evaluator.py`
   实现双框一对一融合、候选池、确定性最高分选择、各轮成功判断和最终四级证据。
8. `src/instatarget/controller/state_machine.py`
   按第 8 节实现纯跨帧转移，增加单框找回连续确认，移除“每个 attempt 都 transition”的用法。
9. `src/instatarget/controller/depth_aware_track_controller.py`
   按 startingMode 生成不同路线，只在最终轮原子提交一次，并正确处理 motion/template/cooldown。
10. `src/instatarget/app/driver.py` 及线程消息路径
    允许最多三个 attempt，但根据 controller 返回值自然结束；仍保证每个输入帧只写一个输出。
11. tests 与现有文档
    完成第 13 节测试后，同步 `docs/modules/controller.md`、`docs/process.md`、`docs/interface.md` 和旧 V2
    文档中与本计划冲突的描述。

## 13. 必须覆盖的测试

### 13.1 FOV 与布局

- 初始化及所有状态、所有轮次的 ViewSpec 都严格为 `120° × 120°`。
- 四角视图恰好 4 张、无中心图、局部偏移为 `±40°`，中心共同区域位于 seed。
- 跨经线和近极点 seed 不产生无效 pitch 或不对称布局。
- cubemap 恰好 6 张，方向固定，覆盖全景且不受 seed/phase 影响。

### 13.2 OverlapThreshold 与融合

- 配置默认 `OverlapThreshold=0.70` 并能被其他合法值替换。
- 配置默认 `FusionSourceMinConfidence=0.80`；`min(a,b)==0.80` 必须通过，低于 0.80 必须失败。
- 所有高重合融合和 overlap 输出条件只读取 `OverlapThreshold`，可靠融合还必须读取
  `FusionSourceMinConfidence`，两者都不得重复硬编码。
- 普通第一轮在 `y=0.30` 时不融合，严格大于时融合。
- 后续轮次和 LOST 第一轮在 `y=OverlapThreshold` 时不融合，严格大于时融合。
- OverlapRate 使用交集除以较小框面积，与 IoU 明确区分。
- 同一 viewId 不融合；每个局部框最多参与一个 FuseBox；不存在三框或递归融合。
- 多伙伴冲突选择 overlap 更高的配对，调换输入顺序不改变结果。
- 跨经线候选正确融合，FuseBox 是包含两源框的最小 seam-aware 包围框。
- 源局部框在生成 FuseBox 后仍保留在候选池。
- 源框门控只影响 `RELIABLE_FUSED` 分类和可靠状态更新；不阻止 FuseBox 生成，也不阻止最终轮
  按最高置信度输出。
- 第一轮候选即使 overlap 和融合总分都通过，只要任一源框低于
  `FusionSourceMinConfidence`，就不得提前输出。
- 未通过源框门控但在最终轮分数最高的 FuseBox 仍然输出，并分类为 `WEAK`；不得更新运动历史、
  模板或完成找回。
- 融合置信度公式覆盖边界值和典型值。

### 13.3 分状态轮次

- TRACKING 第一轮失败后只生成第二轮；第二轮低分也直接输出，绝不生成第三轮。
- TRACKING 第二轮弱结果提交后，下一帧状态变成 UNCERTAIN；当前帧不补做第三轮。
- UNCERTAIN/RECOVERING 第二轮超过 SuccessRate 直接结束，未超过时生成第三轮。
- UNCERTAIN/RECOVERING 第三轮直接输出最高候选，绝不生成第四轮。
- LOST 第一轮固定生成 6 张 cubemap，不生成 c1 四角视图。
- LOST 第一轮失败后，以最高候选中心生成 4 张第二轮四角视图。
- LOST 第二轮直接输出最高候选，绝不生成第三轮。
- 各状态最大视图数分别为 8、14、14、10，且所有 viewId 唯一。
- 第一轮最高局部框即使超过 SuccessRate 也不能提前输出。
- 不得跳过最高候选而输出排名较低的合格融合框。

### 13.4 状态转移

- TRACKING 的可靠证据保持 TRACKING；弱或空结果进入 UNCERTAIN。
- UNCERTAIN 的可靠证据返回 TRACKING；WEAK 正确执行 patience；MISSING 直接进入 LOST。
- LOST 的可靠融合直接返回 TRACKING 并重置运动历史。
- LOST 的可靠单框进入 RECOVERING，但 `valid=False` 且不写运动历史。
- RECOVERING 连续单框达到 `recoverConfirmFrames` 才返回 TRACKING。
- RECOVERING 的可靠融合直接完成找回；WEAK/MISSING 返回 LOST。
- 每个输入帧只 transition 一次，计数器只增加一次。
- 弱候选、motion fallback 和未确认找回都不更新模板或运动历史。

### 13.5 空候选、事务和集成

- 第一轮空候选使用 c1 作为第二轮 seed。
- TRACKING/LOST 第二轮空候选立即 motion fallback。
- UNCERTAIN/RECOVERING 第二轮空候选进入第三轮，第三轮空候选 motion fallback。
- 旧 transaction、旧 attempt、重复响应、错误 viewId 和跨帧响应仍抛出 `ProtocolError`。
- 单线程与多线程 driver 对相同输入产生相同逐帧输出。
- RGB-only 和 RGB-D 使用完全相同的视图、轮次、融合和状态转移结构。

## 14. 完成定义

只有满足以下全部条件才算实施完成：

- 初始化及所有搜索 ViewSpec 固定使用 120° 最大视域，不存在逐渐放大的生产路径。
- `OverlapThreshold` 是唯一高重合阈值来源，默认 0.70，并控制融合、输出资格和可靠融合分类。
- TRACKING 为两轮、UNCERTAIN/RECOVERING 为三轮、LOST 为“六面第一轮 + 四角第二轮”。
- 每个状态都在规定的最终轮直接结束，不存在隐藏的额外 attempt。
- FuseBox 最多包含两个框，显式标记融合状态，使用指定置信度公式和 seam-aware 最小包围框。
- 最终证据和跨帧状态转移按第 7、8 节执行，找回单框需要连续确认，FuseBox 还必须通过
  `FusionSourceMinConfidence` 才能作为可靠融合。
- 前几轮不污染跨帧状态；每帧只做一次 transition、一次原子提交和一次结果发布。
- 配置、YAML、driver、类型、测试和关联文档全部同步。

## 15. 已明确的边界决策

1. ViewSpec 的 `1/3` 指局部单轴角覆盖比例，因此中心间距为 80°、相对 seed 偏移为 ±40°；预测框
   OverlapRate 是另一套基于框面积的定义。
2. LOST 第一轮虽然采用与其他状态第三轮相同的 cubemap 布局，但它不是最终轮；可靠融合可以提前
   输出，否则继续以最高候选中心进行第二轮局部确认。
3. TRACKING 的弱第二轮结果先输出并转入 UNCERTAIN，第三轮从下一输入帧才开始启用。
4. 最终轮强制输出和可靠测量是两个概念；弱框可输出，但不能更新运动历史或模板。
5. 初始化模板也按“所有局部视域最大”固定为 120°，实现后需要重点评估目标像素占比变化。
6. 空候选 fallback 只用于补全未定义的空集合情况，不能覆盖存在候选时的最高分输出规则。
