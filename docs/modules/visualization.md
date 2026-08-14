# 可视化模块

可视化是诊断输出层，不是跟踪决策层。`VisualizationRecorder` 接收已经生成的帧、局部视图、后端观测和几何投影，只写 PNG 文件，不修改输入数组、控制器状态或结果对象。

## 阶段

配置允许四个阶段：

| 阶段 | 内容 |
|---|---|
| `local_rgb` | 每个局部视图的 RGB 裁剪 |
| `depth_rgb` | 深度预处理器生成的伪彩色图 |
| `backend_box` | 局部 RGB 上绘制后端框和融合分数 |
| `geometry_box` | 原始 ERP RGB 上绘制回投影框和融合分数 |

路径格式为：

```text
<outputRoot>/<sequence>/frame_<六位帧号>/<stage>/view_<四位视图号>.png
```

`ResultVisualizationRecorder` 另写每帧一张最终 ERP 图，在 `resultVisualRoot/frame_<六位帧号>.png` 中绘制已提交框和状态分数。

## 对测试的影响

可视化阶段不会向 HiT、融合头、控制器或结果 sink 回传任何值，因此不会影响测试精度、状态转换和 BFoV 输出。它会产生额外图像写入和磁盘占用；比赛容器默认关闭可视化。
