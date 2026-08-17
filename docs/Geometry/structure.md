# Geometry 模块结构

Geometry 负责 ERP、球面方向、BFoV、局部透视图和循环 bbox 之间的全部转换。

| 文件 | 职责 |
|---|---|
| `projection_math.py` | 球面、向量、ERP 像素与透视射线数学 |
| `bfov_projector.py` | ERP 到 LocalView 的采样 |
| `spherical_geometry.py` | bbox/BFoV 转换及局部边界一次回投门面 |
| `seam.py` | ERP 经线跨越和循环区间 |

深入阅读：[viewTypes.md](viewTypes.md)、[coordinateTransforms.md](coordinateTransforms.md)、[perspectiveProjection.md](perspectiveProjection.md)、[seamHandling.md](seamHandling.md)、[parameters.md](parameters.md)。
