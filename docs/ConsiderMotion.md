# MotionPredictor 与 SingleScore 升级方案

> 本文仅描述建议，不修改现有 motion、score calibration、数据类型或运行流程代码。

## 目标

升级后的运动模块需要同时完成两件事：

1. 根据前几帧可靠测量预测当前帧目标中心、尺寸、可选深度及其不确定度。
2. 对每一个局部框计算“它与运动预测有多一致”的 `motionScore`，再与后端外观分数合成为该单框最终使用的 `SingleScore`。

目标分数语义建议固定为：

```text
backend fusedScore --外观校准--> appearanceProbability
motion residual   --运动校准--> motionProbability
SingleScore = 0.70 * appearanceProbability + 0.30 * motionProbability
```

`SingleScore` 才进入 StateEvaluator 的单框排序和两框融合。这里的 70/30 是第一版实验参数，不应被当作最终标定结果。

## 当前实现的问题

现有 `SphericalMotionEstimator` 已经具有球面切平面、置信度加权、Huber 重加权、尺度/深度趋势和重捕获重置，基础方向是合理的。主要缺口在“预测不确定度如何转成每个候选框的概率”。

当前 `_motionScore()` 近似为：

```text
motionScore = (cos(angularDistance) + 1) / 2 * predictionConfidence
```

当候选都位于预测中心附近时，余弦值会集中在很窄的高分区间。例如角误差从 5 度增加到 20 度，余弦映射仍非常接近 1。这样即使最终写成 30% 权重，运动项的实际区分贡献也可能远低于 30%。此外，它没有按当前预测不确定度归一化：相同 10 度误差在稳定跟踪和长时间丢失后不应得到相同评价。

## 推荐的 MotionPredictor

### 状态表示

建议采用“球面局部切平面上的自适应常速度 Kalman Filter”，并保留现有 Huber 逻辑作为异常测量保护。状态可分为：

```text
[east, north, vEast, vNorth,
 logWidth, logHeight, vLogWidth, vLogHeight,
 logDepth, vLogDepth]
```

- `east/north` 是相对当前参考球面点的切向坐标，避免直接对跨经线 yaw 做差。
- 尺寸与深度继续使用 log 空间，避免出现负值，并让比例变化接近线性。
- 没有深度时只启用前 8 维，不应把缺失深度当作 0 测量。
- 状态转移矩阵使用实际 `dt`，过程噪声随 `dt`、速度和当前 TrackMode 调整。

每次更新后把参考切平面重新放到最新可靠中心，并通过三维单位向量转换状态，避免目标移动较远后切平面线性化失效。

### 协方差与异常值

预测必须输出协方差，而不只输出一个 `angularUncertaintyRad` 标量。测量噪声建议根据以下因素放大：

- `SingleScore` 较低；
- 框接近局部视图边缘；
- 框尺寸很小；
- 深度有效比例低；
- 当前处于 RECOVERING/LOST。

候选创新使用 Mahalanobis distance。Deep SORT 的 Kalman 实现同样使用 bbox 状态、预测协方差和卡方阈值进行 gating；它适合作为实现参考，但本项目必须把平面 `(x,y)` 换成球面切向残差。对突然转向或遮挡，参考 OC-SORT 的 observation-centric 思路：不要让长期外推持续主导，重新获得可靠观测后用最近观测速度重建或重置运动状态。

## MotionScore 暂定算法

对候选框构造残差：

```text
rCenter = [eastResidual, northResidual]
rScale  = [log(candidateWidth/predictedWidth),
           log(candidateHeight/predictedHeight)]
rDepth  = log(candidateDepth/predictedDepth)  # 仅双方深度有效时
```

用预测协方差与测量协方差之和 `S` 归一化：

```text
d2Center = rCenter^T inv(SCenter) rCenter
d2Scale  = rScale^T  inv(SScale)  rScale
d2Depth  = rDepth^2 / depthVariance
d2 = d2Center + lambdaScale*d2Scale + lambdaDepth*d2Depth
```

第一版建议：

```text
rawMotionScore = exp(-0.5 * min(d2, d2Max))
```

也可以在协方差经过验证后使用 `chi2.sf(d2, degreesOfFreedom)`。SciPy 将 `sf` 定义为 `1-cdf`，它可以把按卡方分布解释的创新距离转成一致性分数。自由度必须随启用的中心、尺度、深度维数变化。

运动历史不足或预测已明显退化时，不应返回 1.0。建议返回中性值 0.5，并同时输出 `motionReliability=0`；可靠性恢复后再逐渐增加运动项影响：

```text
effectiveMotion = reliability * calibratedMotion + (1-reliability) * 0.5
```

## MotionScore 分布拉伸

不要对“当前一帧的候选”做 min-max normalization。那会让每帧必然出现一个 0 和一个 1，破坏跨帧、跨轮阈值的可比较性。

正式做法是在独立 calibration dataset 上收集：

- 原始 `d2`、`rawMotionScore`；
- 候选与真值的球面中心误差和 IoU；
- 候选是否为正确目标的标签；
- TrackMode、预测间隔、目标纬度和局部视图边缘距离。

然后单独拟合 motion calibration。样本足够时优先比较 Beta Calibration 与 isotonic regression；样本不足时先使用固定的 robust logit-temperature 映射：

```text
z = (logit(rawMotionScore) - validationMedian) / max(validationIQR, epsilon)
calibratedMotion = sigmoid(z / temperature)
```

`validationMedian`、`validationIQR` 和 `temperature` 必须由训练/校准集冻结，不能逐帧计算。目标是让正确候选和错误候选覆盖足够宽的分数范围，并通过 reliability diagram、Brier score 和候选排序 AUC 验证，而不是只追求视觉上“分数更分散”。

## ScoreCalibration 模块边界与命名

运动分数要等局部框回投后才能计算，而当前外观 Beta Calibration 发生在回投前。因此“都放在 scoreCalibration 中”建议理解为统一由同一模块拥有公式和参数，但调用分两步：

1. `calibrateBackendFusedScore()`：backend 后、回投前执行。
2. `calibrateMotionScore()` 和 `composeSingleScore()`：回投得到中心、尺寸残差后执行。

建议字段语义：

| 字段 | 含义 |
|---|---|
| `backendFusedScore` | backend 原始 RGB/深度/上下文融合分数 |
| `appearanceProbability` | Beta Calibration 后的外观概率 |
| `rawMotionScore` | Mahalanobis residual 映射结果 |
| `motionProbability` | 分布校准后的运动概率 |
| `singleScore` | 70% 外观 + 30% 运动，供 StateEvaluator 使用 |

不要把后端原始分数和最终 `SingleScore` 都继续叫 `fusedScore`，否则日志无法判断是哪一层出了问题。

## 第一版 SingleScore

第一版直接采用可解释的线性池：

```text
SingleScore = clip(
    0.70 * appearanceProbability
  + 0.30 * effectiveMotion,
    0, 1
)
```

只有在两个输入都校准到相近概率尺度后，30% 才有接近理论的影响。第二阶段可比较 log-odds pooling：

```text
SingleScore = sigmoid(
    bias
  + 0.70 * logit(appearanceProbability)
  + 0.30 * logit(effectiveMotion)
)
```

log-odds 版本对极高/极低证据更敏感，但必须重新标定 `successRate`、`fusionSourceMinConfidence` 和双框融合后的概率，不能直接替换上线。

## 实施顺序

1. 先只记录候选残差、预测协方差和现有分数，不改变决策。
2. 离线实现新 `motionScore`，检查正确/错误候选分布是否真正分离。
3. 冻结 motion calibration 参数，并保留旧外观分数、运动分数、SingleScore 三列日志。
4. 以 70/30 线性池做 shadow evaluation。
5. 联合扫描 80/20、70/30、60/40，以及 `successRate` 和来源最低分阈值。
6. 最后再决定是否将 robust regression 完全替换为 Kalman，或采用二者混合。

必须比较 mean IoU、成功率、每状态帧数、重捕获误检率、候选排序 AUC、Brier score、最差延迟，并按目标纬度、速度、遮挡和视图边缘距离分组。

## 参考资料

- [Deep SORT Kalman filter](https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py)：常速度 bbox 状态、预测协方差、Mahalanobis gating 和卡方阈值。
- [OC-SORT](https://github.com/noahcao/OC_SORT)：针对遮挡和非线性运动的 observation-centric 修正思路。
- [Beta Calibration Python](https://github.com/betacal/python)：现有外观校准方法的官方实现参考。
- [SciPy chi-square distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)：`cdf/sf/ppf` 定义和卡方概率映射。

