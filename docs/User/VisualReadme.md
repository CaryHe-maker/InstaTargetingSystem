# 可视化输出说明

运行命令提供 `--mid-visual-root` 和 `--result-visual-root` 两个输出选项。快速命令已经为两者指定了 `midVisual` 与 `result\visualResult`。

## 输出内容

`midVisual` 按帧和视图写入四类 PNG：

- `local_rgb`：局部 RGB 输入。
- `depth_rgb`：深度伪彩色输入；RGB-only 没有深度处理阶段，因此不会产生有效深度图。
- `backend_box`：HiT 局部框及融合分数。
- `geometry_box`：回投影到 ERP 的框及融合分数。

`result\visualResult` 每帧写一张 ERP 图，绿色框表示已提交的结果，标签包含控制器状态分数。

## 影响范围

图像由 `VisualizationRecorder` 和 `ResultVisualizationRecorder` 在结果生成后写入。它们不修改模型输入、观测、状态、模板或结果 sink，因此不会影响测试值。关闭可视化只减少文件写入和运行开销。
