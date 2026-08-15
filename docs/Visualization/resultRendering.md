# 最终结果绘制

## 输入

`ResultVisualizationRecorder.record()` 接收原 ERP FramePacket、最终 TrackResult 和可选 StateObservation 分数。它只显示 Controller 已提交结果，不重新选择候选。

## 框绘制

普通 bbox 画一个矩形；跨缝 bbox 通过 `splitSeamBox()` 在右边和左边分别画一段。两段共用标签，逻辑上仍是一个目标。坐标在绘制前裁剪，防止负坐标或超宽框写出数组边界。

## 标签

标签可包含置信度、状态、valid 或 stateScore。绘制函数使用固定荧光绿、线宽和标签间距；这些是代码常量，不是算法超参数。

## 解读顺序

最终图异常时，应按 geometry_box → backend_box → local_rgb 逆向检查。最终框正确但 valid=False 通常表示输出来自弱观测或运动预测，不是绘图错误。

## 性能边界

最终图在处理计时停止后写入。关闭结果可视化能减少总进程耗时和磁盘占用，但不会改变 `time.json` 或跟踪数值。

