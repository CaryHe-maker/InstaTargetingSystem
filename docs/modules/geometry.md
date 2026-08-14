# 几何模块

几何模块将 ERP 图像坐标、球面 BFoV 和局部透视图相互转换，是全景跟踪与普通局部跟踪器之间的边界。

## 转换接口

- `bboxToBfov()`：将 ERP 框的边界采样投影到单位球面，生成中心和水平/垂直视场角。
- `bfovToBbox()`：将球面 BFoV 的边界采样回投影到 ERP，得到可跨水平接缝的框。
- `cropViews()`：按 `ViewSpec` 将 ERP RGB 和可选深度重采样为固定尺寸局部视图。
- `localBoxToBfov()`：将 HiT 局部像素框转换回球面 BFoV。

## ERP 接缝

水平方向采用周期坐标，垂直方向保持有限范围。`wrapHorizontal` 绘图和 `minimalCircularInterval` 目标框计算都遵循同一接缝约定。局部视图裁剪会在水平边界处拼接源图像，保证模型看到连续的目标区域。

## 数值约束

所有方向向量必须接近单位长度；经纬角使用弧度，输出比赛文本时才转换为角度。视场角满足配置限制，框宽高始终为正数。边界采样数由 `geometry.boundarySamplesPerEdge` 控制，默认值为 65。
