# 坐标转换链

## ERP 像素到球面点

`erpPixelToSphericalPoint()` 将 x/W 映射到 [-π,π) yaw，将 y/H 映射到 [π/2,-π/2] pitch，再计算单位向量。反向 `sphericalPointToErpPixel()` 对 yaw 做 wrap，对 pitch 做 clamp。

## 球面点到相机坐标

`cameraBasis()` 根据 BFoV 中心构造 forward/right/up，并应用 roll。局部像素先用 FOV 和焦距换算成相机平面偏移，再组合三个基向量得到世界单位射线。

焦距关系为：

```text
f = imageExtent / (2 * tan(FOV/2))
```

所以相同像素偏移在不同 FOV 下对应不同角度，不能用简单的像素比例把 HiT bbox 映回 ERP。

## 局部框到 BFoV

局部框的边界点通过相机射线映射到球面。Geometry 对每条边采样 `boundarySamplesPerEdge` 个点，而不是只看四角；这是因为透视矩形边投影到 ERP 后通常是曲线。所有边界方向的最小球面/循环覆盖形成目标 BFoV 和 ERP bbox。

## ERP bbox 到 BFoV

初始化框中心先转成球面点，水平/垂直像素跨度按 ERP 角分辨率转成角尺寸。跨缝框的中心使用循环区间，不使用普通 `(x1+x2)/2`。

## 数值边界

yaw 必须 wrap，pitch 必须 clamp，向量必须重新归一化，FOV 必须位于 (0,π)。极点附近优先使用向量计算，避免 yaw 在一个像素移动下大幅变化。

