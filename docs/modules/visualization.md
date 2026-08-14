# Visualization

可视化是 app 层的只读旁路，不参与 HiT 推理、分数计算或控制决策。recorder 保存 backend 实际送入 HiT 的局部图，因此诊断图与模型输入逐像素一致。

| stage | 内容 |
|---|---|
| `local_rgb` | backend 实际使用的局部 RGB。RGB-only 为原图；RGBD 为深度边缘增强后的 RGB。 |
| `depth_rgb` | RGBD 深度边缘预测图：白色表示边缘，黑色表示非边缘。RGB-only 不产生有效深度图。 |
| `backend_box` | 在与 HiT 输入完全相同的局部 RGB 上绘制候选框与 `fuseScore`。 |
| `geometry_box` | 在原始 ERP RGB 上绘制回投影框与 `fuseScore`。 |

RGBD 的 `local_rgb` 与 `backend_box` 都来自 `TrackerBackend.lastPreparedViews`；不会重新生成伪彩色深度图，也不会绘制第二个模型结果。PNG 写出保持无损 `uint8 [H,W,3]`。

目录结构：

```text
<root>/<sequence>/frame_000000/
  local_rgb/view_0000.png
  depth_rgb/view_0000.png
  backend_box/view_0000.png
  geometry_box/view_0000.png
```
