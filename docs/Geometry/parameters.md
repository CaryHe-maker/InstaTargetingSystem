# Geometry 参数索引

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `geometry.viewWidthPx` | 256 | LocalView 输出宽度和 HiT 搜索输入尺寸 |
| `geometry.viewHeightPx` | 256 | LocalView 输出高度 |
| `geometry.boundarySamplesPerEdge` | 65 | bbox/BFoV 边界每边采样点数 |
| `geometry.minFovDeg` | 20 | 运动回退包络最低 FOV |
| `geometry.maxFovDeg` | 120 | 所有搜索 ViewSpec 的固定水平/垂直 FOV |

`maxFovDeg` 由 schema 强制为 120。四角中心水平/垂直偏移 40 度和 cubemap 六个方向目前是 `controller/recovery_planner.py` 中的布局常量，不是 YAML 参数。

提高局部分辨率会近似按像素数增加投影和模型成本；HiT 当前仍缩放到 256×256，因此只改 ViewSpec 尺寸不会自动增加网络有效分辨率。提高 boundarySamples 主要增加框回投成本。优化时应分别测试目标位于视图中心、边缘、极点和 ERP 经线附近的直接 bbox、BFoV 误差与 `envelopeInflation`。

