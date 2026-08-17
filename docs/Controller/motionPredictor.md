# 球面运动预测算法

实现位于 `controller/motion_estimator.py::SphericalMotionEstimator`。目标是输出下一帧搜索中心 c1、目标角尺寸、深度趋势和不确定度。

## 样本选择

历史是长度为 `windowLength` 的可靠测量 deque。预测帧和未接受的弱候选不会加入历史。每个样本保存球面中心、时间戳、角尺寸、置信度和可选深度。

## 角速度拟合

算法在最新球面点建立 east/north 正交切平面，把历史单位向量投影到二维局部坐标。每个轴拟合“截距 + 时间 × 速度”，基础权重取样本置信度，随后执行三轮 Huber 重加权降低离群点影响。

速度模长被 `maxAngularSpeedRadPerSec` 裁剪；若残差超过 `maxTangentSpanRad`，认为窗口无法被单个局部线性模型解释，速度退化为 0。切平面基对极点有专门回退，不直接对 yaw 做差，因此不会在 ±180 度经线处跳变。

## 球面外推

预测时将二维切向速度乘以时间差，加到最新单位向量后重新归一化，再转换回 yaw/pitch。它是短时间常速度近似，不是完整大圆指数映射，因此 `maxPredictionHorizon` 和速度上限共同限制外推距离。

## 尺度与深度

水平/垂直角尺寸在 log 空间分别做线性拟合，再把 log rate 裁剪到 `maxLogScaleRatePerSec`。深度也在 log 空间拟合，仅使用有效且有置信度的样本。没有深度时保留角运动预测，并在 `degradedReasons` 标记 `missing_depth`。

## 协方差与可靠性

角不确定度由拟合残差、`processNoiseRadPerSec × dt` 和有限样本项组成；预测置信度随时间指数衰减。`MotionPrediction` 同时输出 2×2 中心协方差、2×2 log 尺度协方差和可选深度方差。当前第一版使用由标量不确定度构成的对角协方差，接口允许后续 Kalman 实现提供非对角项。

少于 `minSamplesForVelocity` 个样本时速度仍退化为 0，但最新测量仍可作为位置/尺度锚点。可靠性按 `min(1, sampleCount/minSamplesForVelocity)` 连续增长，再由预测置信度和角不确定度衰减，因此不会因为样本数开关长期停在中性 0.5。`degradedReasons` 在速度样本不足时仍保留 `insufficient_motion_samples`，表示此时可用的是降权位置先验而不是成熟速度模型。

控制器把运动启动与公开测量提交解耦：初始化后的首个非空候选即使没有通过状态提交，也只会向运动窗口写入一次有界低置信度启动样本，不更新公开 bbox/BFoV。正常 backend 每轮产生候选时，第三个输入帧开始前窗口已有两个时间点，速度和运动残差评分可以工作；后续弱候选不会持续写入并污染窗口。RecoveryPlanner 用角不确定度扩展运动回退包络，但搜索 ViewSpec 本身仍固定 120 度。

## 局部视图运动分数

用于 SingleScore 的 `effectiveMotion` 是同一帧内的空间搜索先验，不再使用检测框回投中心。每个局部图以自己的 `ViewSpec.bfov.center` 与该帧 MotionPredictor 的预测中心计算球面最短大圆夹角：

```text
angle = acos(clamp(dot(viewCenter, predictedCenter), -1, 1))
effectiveMotion = clip(1 - angle/(30deg)*0.1, 0, 1)
```

因此预测中心为 1.0，30 度为 0.9，45 度为 0.85，60 度为 0.8，90 度为 0.7，球面对跖点 180 度为 0.4；区间内连续线性下降。同一帧同一个局部图中的候选共享该视图分数，不进行帧间归一化，也不按当前候选集合做 min-max。

原来的中心/尺度/深度协方差残差函数仍保留为离线诊断能力，但不再作为 Runtime 的 `effectiveMotion` 或 SingleScore 输入。

## 重捕获

`resetFromMeasurement()` 清空旧窗口，只放入重捕获测量，避免丢失前速度把下一帧再次推离目标。优化时应重点绘制预测中心误差、残差、速度裁剪频率和历史有效样本数。

