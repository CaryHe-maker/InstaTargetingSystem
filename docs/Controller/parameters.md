# Controller 参数索引

当前值来自 `configs/RGBonly.yaml` 和 `configs/RGBD.yaml`。两份配置的 Controller 参数一致。

## StateEvaluator

| 参数 | 当前值 | 生产路径作用 |
|---|---:|---|
| `evaluator.successRate` | 0.90 | 仅复制到 `StateObservation` 诊断字段 |
| `evaluator.firstRoundFusionOverlap` | 0.30 | schema/配置兼容；当前生产评估不读取 |
| `evaluator.overlapThreshold` | 0.70 | schema/配置兼容；生产 Fusor 使用代码常量 0.70 |
| `evaluator.fusionSourceMinConfidence` | 0.80 | 融合候选被接受时两个来源各自的最低 SingleScore |
| `evaluator.supportWeight` | 0.25 | schema/接口兼容；当前生产评估不读取 |
| `evaluator.agreementWeight` | 0.25 | schema/接口兼容；当前生产评估不读取 |
| `evaluator.minReacquireViews` | 2 | schema/旧重捕获兼容；当前生产评估不读取 |

当前只有 `fusionSourceMinConfidence` 直接影响测量接受。Fusor 对每一轮都使用固定的 0.70 overlap 常量枚举全部候选对；来源最低分不会阻止候选生成或排序，只会阻止低来源分的融合候选被接受为测量。

## 状态、事务和模板

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `tracking.stableFramesBeforeUpdate` | 8 | 在线模板更新前的稳定帧数 |
| `tracking.sameFrameEscalationEnabled` | true | 允许 TRACKING/UNCERTAIN 在第一轮后进入第二轮 |
| `tracking.maxAttemptsPerFrame` | 2 | 固定同帧最多两轮；LOST 使用一次 12 视图恢复计划 |
| `tracking.maxViewsPerFrameTotal` | 12 | 单帧事务总视图预算 |
| `tracking.reacquireCooldownFrames` | 2 | 重捕获后模板更新冷却 |
| `tracking.maxPredictionHorizon` | 3 | Controller 请求的最大运动外推帧数 |

## 运动预测

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `tracking.windowLength` | 5 | 可靠运动样本窗口长度 |
| `motion.minSamplesForVelocity` | 2 | 开始拟合速度所需样本数 |
| `motion.maxTangentSpanRad` | 1.20 | 拟合残差过大时退化为零速度的界限 |
| `motion.huberDeltaRad` | 0.15 | Huber 重加权阈值 |
| `motion.processNoiseRadPerSec` | 0.04 | 随预测时间增长的不确定度 |
| `motion.maxAngularSpeedRadPerSec` | 2.0 | 切向角速度上限 |
| `motion.maxLogScaleRatePerSec` | 1.0 | 目标角尺寸对数变化率上限 |

`SphericalMotionEstimator` 还提供构造参数 `alpha=0.70`、`beta=0.20`，供旧 alpha-beta 兼容接口保留；当前详细窗口拟合路径不使用它们更新位置。

## 分数常量

当前 SingleScore 权重 0.70/0.30、运动尺度权重 0.35、深度权重 0.15、最大 d2=25、中心测量标准差 0.025 rad 和 log 尺度测量标准差 0.08 位于 `controller/fused_score.py`。它们尚未进入严格 YAML schema；替换前必须完成独立校准并联动重标 `tracking.candidateMinScore` 与来源最低分。

## 回退包络和兼容参数

| 参数 | 当前值 | 状态 |
|---|---:|---|
| `tracking.contextScale` | 2.0 | 扩展预测回退 BFoV |
| `tracking.contextMarginRatio` | 0.15 | 回退 BFoV 额外边距 |
| `tracking.candidateMinScore` | 0.40 | 所有最佳候选被接受为测量的最低 confidence |
| `tracking.scaleClusterTolerance` | 0.50 | 旧尺度聚类容差 |
| `tracking.guardYawStepDeg` | 120 | 旧 guard 视图步长 |
| `tracking.minViewsForCommit` | 2 | 旧聚合最少视图数 |
| `tracking.uncertainFovScale` | 1.25 | 当前固定 120 度搜索不使用 |

除 `candidateMinScore` 外，本节其余 tracking 字段为回退包络或旧聚合兼容项。RecoveryConfig 中的 `maxViewsPerFrame=12`、`globalSearchInterval=5`、`ringRadii=[1,1.75,2.5]`、`viewsPerRing=[4,8,12]`、`cubeMapOverlapRatio=0.10`、`maxCoveredCells=256` 保留恢复内存/旧环搜配置；当前实际 LOST 路径固定生成两个旋转 cubemap（12 张），不运行环搜。`decisionGate.*` 也只受 schema 校验，生产 StateEvaluator 会忽略整组配置。

## ScoreGroup 阈值

状态机只保存最近 10 个已提交 `StateScore`。历史少于 2 个时不计算阈值；第三个状态决策使用第二个分数与第一个分数比较。历史为 2 至 9 个时，`UT=0.5*max+0.5*min`、`LT=0.2*max+0.8*min`；达到 10 个后，降序第 5 大为 UT、第 8 大为 LT。阈值只用于下一帧状态选择，是否把当前候选写入运动历史仍由独立的 measurement acceptance gate 决定。

## 调参顺序

先校准单框分数，再选择 `tracking.candidateMinScore` 和 `fusionSourceMinConfidence`；随后评估固定 0.70 overlap 常量是否需要代码级改动；最后调整状态和视图预算。`successRate`、`firstRoundFusionOverlap`、`overlapThreshold` 或 `decisionGate.*` 的 YAML 值当前不会改变生产决策，不能把修改这些值造成的实验波动解释为算法效果。

