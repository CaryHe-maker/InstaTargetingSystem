# 中间阶段产物

`VisualizationRecorder` 提供四个独立阶段，用同一 frameIndex/viewId 命名，便于逐层定位误差来源。

## local_rgb

保存 Geometry 实际送入 HiT 的局部 RGB。用于检查视域中心、实际 FOV 覆盖、透视拉伸和目标是否出现；TRACKING 为 30°–120°动态 Type1，UNCERTAIN/LOST 使用固定 120°视图。若此处没有目标，后端低分通常不是模型问题。

## depth_rgb

保存 DepthPreprocessor 的三通道伪彩色。用于检查有效 mask、远近亮度、边缘增强和 RGB/深度空间对齐。该产物是诊断输出，不会再次送回 backend。

## backend_box

在每个 LocalView 上绘制 HiT 局部框。标签 `fuseScore=raw/appearance` 依次显示 backend 原始融合分与 Beta Calibration 后的外观概率，可区分“模型框错”“外观校准错”和“Geometry 回投错”。

## geometry_box

把 ProjectedObservation 的直接 ERP bbox 绘制在原帧上，验证局部框到球面/ERP 的转换和跨缝语义。标签 `score=single/effectiveMotion/appearanceProbability/inflation` 依次显示最终 SingleScore、可靠性混合后的运动概率、Beta Calibration 后的外观概率和间接/直接 bbox 面积膨胀比。

## 多轮组织

Runtime 在处理阶段只把每轮数据保存在 `visualizationBatches`；计时停止后按 round/viewId 写图。四阶段可以通过 `visualization.stages` 独立启用，关闭阶段不得创建空目录。

