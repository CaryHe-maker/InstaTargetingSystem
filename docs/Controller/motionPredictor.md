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

## 不确定度

角不确定度由拟合残差、`processNoiseRadPerSec × dt` 和有限样本项组成；预测置信度随时间指数衰减。RecoveryPlanner 用不确定度扩展运动回退包络，但搜索 ViewSpec 本身仍固定 120 度。

## 重捕获

`resetFromMeasurement()` 清空旧窗口，只放入重捕获测量，避免丢失前速度把下一帧再次推离目标。优化时应重点绘制预测中心误差、残差、速度裁剪频率和历史有效样本数。

