# 跟踪指标算法

## 平面 IoU

`bboxIoU()` 用交集面积除以并集面积，适合不循环的局部或普通图像框。ERP 结果跨经线时必须使用 `circularBBoxIoU()`：先把每个框拆成最多两个水平段，累加段对交集，再计算并集。

StateEvaluator 的 OverlapRate 也使用交集，但分母是较小框面积，用于判断两个预测是否属于同一目标；Evaluation IoU 分母是并集，用于衡量预测与真值的形状吻合度。二者不可直接比较阈值。

## 成功曲线和 AUC

`successCurve()` 在 0 到 1 的 21 个 IoU 阈值上计算 `IoU > threshold` 的帧比例。AUC 对成功曲线做梯形积分；`successRate@0.5` 是其中一个工作点，`meanIoU` 是逐帧均值。

## 跟踪丢失率

丢失率用于统计目标可见时预测与真值完全没有重叠的概率：

```text
lostFrameCount = sum(visible frame circular ERP IoU <= 1e-12)
trackingLossRate = lostFrameCount / evaluatedVisibleFrames
```

第 0 帧给定初始化不计分，目标不可见帧不进入分子或分母。`1e-12` 仅吸收投影浮点噪声；它不把“小 IoU”误判为完全丢失。该指标目前只写入单序列和聚合报告，不触发 `LOST`、恢复搜索、运动历史清空或任何输出决策。后续找回算法应以这一基线分层分析，但必须另行设计和 A/B。

## 球面中心误差

两个 BFoV 中心转为单位向量，点积裁剪到 [-1,1] 后取 arccos，得到大圆角距离。它不会在 yaw 跨 ±180 度时产生假大误差。

## 球面 BFoV IoU

`bfovSphericalIoU()` 在 yaw/pitch 网格中采样球面点，分别判断是否落入两个 BFoV。每个纬度样本按 cos(pitch) 加权，补偿 ERP 在极点过采样；加权交集除以加权并集。

采样数 `samplesYaw/samplesPitch` 越高越精确但更慢。它们是函数参数，不是运行 YAML 超参数。比较实验时必须固定采样密度。

## 分数与回投诊断

分数实验除最终 IoU 外必须报告 `appearanceProbability` 和运动概率的 Brier score/reliability diagram、候选排序 AUC，以及按状态和 prediction horizon 分组的结果。回投实验应同时报告中心角误差、BFoV 宽高相对误差、直接 ERP bbox IoU 和 `envelopeInflation`，并按纬度、局部归一化半径与 FOV 分组。

## 可见性和第 0 帧

第 0 帧是给定初始化，不应重复当作模型预测计分。目标不可见帧必须按评测协议明确处理，不能因为真值缺失而让预测和真值序列错位。

所有真实 IoU、丢失率、校准与回归必须基于 `E:\NewDownload\train\manifest.jsonl`。仓库内 `data/` 的小样本只允许做确定性单元测试，不能进入真实指标汇总。

