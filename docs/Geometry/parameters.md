# Geometry 参数索引

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `geometry.viewWidthPx` | 256 | LocalView 输出宽度和 HiT 搜索输入尺寸 |
| `geometry.viewHeightPx` | 256 | LocalView 输出高度 |
| `geometry.boundarySamplesPerEdge` | 65 | bbox/BFoV 边界每边采样点数 |
| `geometry.minFovDeg` | 20 | 运动回退包络最低 FOV |
| `geometry.maxFovDeg` | 120 | cubemap 与 UNCERTAIN 两轮四角的固定 FOV，也是 TRACKING 动态四角的 FOV 上限 |

`maxFovDeg` 由 schema 强制为 120。UNCERTAIN 的 `ViewSpecType1` 把 FOV 下限和上限都设为
该值，因此中心偏移由 FOV 的三分之一得到 40 度。TRACKING 的动态四角由 `ViewSpecType1`
按预测框角尺寸计算 FOV 和偏移，固定使用 30° 下限，并由 `maxFovDeg` 限制上限。

提高局部分辨率会近似按像素数增加投影和模型成本；HiT 当前仍缩放到 256×256，因此只改 ViewSpec 尺寸不会自动增加网络有效分辨率。提高 boundarySamples 主要增加框回投成本。优化时应分别测试目标位于视图中心、边缘、极点和 ERP 经线附近的直接 bbox、BFoV 误差与 `envelopeInflation`。

