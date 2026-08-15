# 中间阶段产物

`VisualizationRecorder` 提供四个独立阶段，用同一 frameIndex/viewId 命名，便于逐层定位误差来源。

## local_rgb

保存 Geometry 实际送入 HiT 的局部 RGB。用于检查视域中心、120 度覆盖、透视拉伸和目标是否出现。若此处没有目标，后端低分通常不是模型问题。

## depth_rgb

保存 DepthPreprocessor 的三通道伪彩色。用于检查有效 mask、远近亮度、边缘增强和 RGB/深度空间对齐。该产物是诊断输出，不会再次送回 backend。

## backend_box

在每个 LocalView 上绘制 HiT 局部框和分数。它展示投影前结果，可区分“模型框错”与“Geometry 回投错”。显示分数是 Beta Calibration 处理后的 LocalObservation 分数。

## geometry_box

把 ProjectedObservation 绘制在 ERP 原帧上，验证局部框到球面/ERP 的转换和跨缝语义。

## 多轮组织

Runtime 在处理阶段只把每轮数据保存在 `visualizationBatches`；计时停止后按 round/viewId 写图。四阶段可以通过 `visualization.stages` 独立启用，关闭阶段不得创建空目录。

