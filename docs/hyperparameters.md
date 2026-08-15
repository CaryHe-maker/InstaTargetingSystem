# 配置与超参数

运行配置采用 `schemaVersion: 1` 的严格 YAML 模式。缺失字段、未知字段、类型错误或越界值会触发 `ConfigError`。

## 与状态搜索直接相关的参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `geometry.maxFovDeg` | `120.0` | 所有初始化、四角和 cubemap `ViewSpec` 的水平/垂直最大视域；必须为 120° |
| `evaluator.successRate` (`SuccessRate`) | `0.90` | 第一轮 FuseBox、UNCERTAIN/RECOVERING 第二轮提前结束和可靠单框判定阈值 |
| `evaluator.firstRoundFusionOverlap` | `0.30` | 普通状态 ROUND_1 生成 FuseBox 的重合率下限 |
| `evaluator.overlapThreshold` (`OverlapThreshold`) | `0.70` | LOST ROUND_1 及所有后续轮次的融合阈值、可靠 FuseBox 可输出阈值 |
| `evaluator.fusionSourceMinConfidence` (`FusionSourceMinConfidence`) | `0.80` | FuseBox 两个源框的单框最低置信度，使用 `>=` |
| `tracking.recoverConfirmFrames` | `2` | RECOVERING 可靠单框连续确认帧数 |
| `tracking.maxAttemptsPerFrame` | `3` | 配置能力上限；实际路线按状态限制为 2/3/3/2 |
| `tracking.maxViewsPerFrameTotal` | `14` | 单帧视图总预算；三轮路线需要至少 14 |

`OverlapRate` 不是 IoU，而是预测框交集面积除以较小预测框面积；跨 ERP 经线时必须使用 seam-aware 计算。所有阈值比较均为严格 `>`，只有源置信度门控为 `>= 0.80`。

## 状态路线

| 起始状态 | 搜索路线 |
|---|---|
| `TRACKING` | 四角 4 + 四角 4，第二轮结束 |
| `UNCERTAIN` | 四角 4 + 四角 4 + cubemap 6 |
| `RECOVERING` | 四角 4 + 四角 4 + cubemap 6 |
| `LOST` | cubemap 6 + 四角 4，第二轮结束 |

第一轮四角 seed 是多帧预测中心 `c1`；第二轮 seed 是第一轮候选最高置信度框中心。四角中心相对 seed 偏移 `±40°`，相邻视域覆盖重合 `1/3`，四视域共同重合区域为 `1/9`。

## 兼容字段

`tracking.uncertainFovScale`、`recovery.ringRadii`、`recovery.viewsPerRing`、`recovery.cubeMapOverlapRatio` 等旧自适应/环搜字段仍由配置模式读取，以保持文件兼容，但不参与新的 `ViewSpec` 生成。不得用这些字段改变固定 120° 视域或状态路线。

其他 `model`、`depth`、`backendFusion`、`fusionHead`、`decisionGate`、`motion`、`runtime` 和 `visualization` 字段按现有配置模式校验。
