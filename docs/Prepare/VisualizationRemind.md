# Visualization 诊断接入说明

> 本文记录诊断图像如何接入可视化模块。原则很简单：应用层负责调用，`visualization` 只负责无损写盘，不参与推理、融合或决策。

---

## 记录内容

当前支持四类诊断图：

- `local_rgb`：几何裁剪后的局部 RGB 视图
- `depth_rgb`：后端已经生成的深度 RGB 图
- `backend_box`：后端局部目标框
- `geometry_box`：回投影到 ERP 原图上的目标框

所有目标框固定使用荧光绿 `#39FF14`，所有输出使用无损 PNG。

## 接入方式

应用层在对应计算完成后调用记录器：

```python
recorder.recordLocalRgb(frame, views)
recorder.recordDepthRgb(frame, depthRgbByViewId)
recorder.recordBackendBoxes(frame, views, localObservations)
recorder.recordGeometryBoxes(frame, projectedObservations)
```

`depthRgbByViewId` 必须是后端已经完成转换的 `viewId -> RGB` 映射，`visualization` 不做任何颜色推断。

## 目录组织

```text
<outputRoot>/
  <sequenceId>/
    frame_000007/
      local_rgb/
      depth_rgb/
      backend_box/
      geometry_box/
```

相同序列、帧号、视图和阶段再次写入时会原子替换旧文件。关闭 `visualization` 后不会创建目录，也不会复制数组。

## 约束

- 输入必须是 CPU `uint8 RGB [H, W, 3]`
- `viewId` 必须稳定对应同一视图
- 局部框与 ERP 框必须遵守现有几何契约
- 诊断图只用于人工检查，不影响跟踪闭环
