# 跟踪指标算法

## 平面 IoU

`bboxIoU()` 用交集面积除以并集面积，适合不循环的局部或普通图像框。ERP 结果跨经线时必须使用 `circularBBoxIoU()`：先把每个框拆成最多两个水平段，累加段对交集，再计算并集。

StateEvaluator 的 OverlapRate 也使用交集，但分母是较小框面积，用于判断两个预测是否属于同一目标；Evaluation IoU 分母是并集，用于衡量预测与真值的形状吻合度。二者不可直接比较阈值。

## 成功曲线和 AUC

`successCurve()` 在 0 到 1 的 21 个 IoU 阈值上计算 `IoU > threshold` 的帧比例。AUC 对成功曲线做梯形积分；`successRate@0.5` 是其中一个工作点，`meanIoU` 是逐帧均值。

## 球面中心误差

两个 BFoV 中心转为单位向量，点积裁剪到 [-1,1] 后取 arccos，得到大圆角距离。它不会在 yaw 跨 ±180 度时产生假大误差。

## 球面 BFoV IoU

`bfovSphericalIoU()` 在 yaw/pitch 网格中采样球面点，分别判断是否落入两个 BFoV。每个纬度样本按 cos(pitch) 加权，补偿 ERP 在极点过采样；加权交集除以加权并集。

采样数 `samplesYaw/samplesPitch` 越高越精确但更慢。它们是函数参数，不是运行 YAML 超参数。比较实验时必须固定采样密度。

## 可见性和第 0 帧

第 0 帧是给定初始化，不应重复当作模型预测计分。目标不可见帧必须按评测协议明确处理，不能因为真值缺失而让预测和真值序列错位。

