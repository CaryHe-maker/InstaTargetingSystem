# Controller 参数索引

当前值来自 `configs/RGBonly.yaml` 和 `configs/RGBD.yaml`。两份配置的 Controller 参数一致。

## StateEvaluator

| 参数 | 当前值 | 生产路径作用 |
|---|---:|---|
| `evaluator.successRate` | 0.90 | 可靠证据和非最终轮提前成功阈值 |
| `evaluator.firstRoundFusionOverlap` | 0.30 | 非 LOST Round 1 生成融合候选的最低 OverlapRate |
| `evaluator.overlapThreshold` | 0.70 | 后续轮融合阈值及可靠融合输出阈值 |
| `evaluator.fusionSourceMinConfidence` | 0.80 | 两个融合来源各自最低置信度 |
| `evaluator.supportWeight` | 0.25 | StateObservation 支持统计的兼容权重 |
| `evaluator.agreementWeight` | 0.25 | StateObservation 一致性统计的兼容权重 |
| `evaluator.minReacquireViews` | 2 | 旧重捕获支持数兼容参数 |

前四项直接影响当前输出。后三项仍受 schema 校验或兼容接口使用，但不改变当前一对一双框融合规则。

## 状态、事务和模板

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `tracking.uncertainPatience` | 2 | UNCERTAIN 中允许连续弱证据的帧数 |
| `tracking.stableFramesBeforeUpdate` | 8 | 在线模板更新前的稳定帧数 |
| `tracking.sameFrameEscalationEnabled` | true | 是否允许同帧进入下一轮 |
| `tracking.maxAttemptsPerFrame` | 3 | 全局轮次上限；状态仍限制为 2/3/3/2 |
| `tracking.maxViewsPerFrameTotal` | 14 | 单帧事务总视图预算 |
| `tracking.reacquireCooldownFrames` | 2 | 重捕获后模板更新冷却 |
| `tracking.recoverConfirmFrames` | 2 | 单框从 RECOVERING 回 TRACKING 的确认帧数 |
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

## 回退包络和兼容参数

| 参数 | 当前值 | 状态 |
|---|---:|---|
| `tracking.contextScale` | 2.0 | 扩展预测回退 BFoV |
| `tracking.contextMarginRatio` | 0.15 | 回退 BFoV 额外边距 |
| `tracking.acceptThreshold` | 0.70 | 旧标量状态接口 |
| `tracking.uncertainThreshold` | 0.45 | 旧标量状态接口 |
| `tracking.recoverAcceptThreshold` | 0.80 | 旧标量状态接口 |
| `tracking.candidateMinScore` | 0.40 | 旧候选门限兼容字段 |
| `tracking.maxRecoveryFrames` | 30 | 旧状态接口恢复期限 |
| `tracking.scaleClusterTolerance` | 0.50 | 旧尺度聚类容差 |
| `tracking.guardYawStepDeg` | 120 | 旧 guard 视图步长 |
| `tracking.minViewsForCommit` | 2 | 旧聚合最少视图数 |
| `tracking.uncertainFovScale` | 1.25 | 当前固定 120 度搜索不使用 |

RecoveryConfig 中的 `maxViewsPerFrame=12`、`globalSearchInterval=5`、`ringRadii=[1,1.75,2.5]`、`viewsPerRing=[4,8,12]`、`cubeMapOverlapRatio=0.10`、`maxCoveredCells=256` 保留恢复内存/旧环搜配置；当前实际全景轮固定六面 cubemap，不运行环搜。

## 调参顺序

先校准单框分数，再选择 successRate；随后选择 overlap 和来源最低分；最后调整状态耐心和视图预算。反过来同时调整多组参数，会无法判断改善来自分数还是更多搜索成本。

