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

`projectLocalBoxBoundary()` 对局部框每条边采样 `boundarySamplesPerEdge` 个点，并只执行一次相机射线到球面的映射。它保留球面/ERP boundary，并由同一组点分别拟合目标 BFoV 和 ERP bbox。

无旋转 BFoV 从边界向量均值中心开始，在局部相机坐标中取水平/垂直角区间；用两个区间中点修正中心，最多迭代四次，最终 FOV 使用 `max-min`，不再使用旧的 `2*max(abs(angle))` 对称扩张。

ERP bbox 不再调用 `bfovToBbox(localBoxToBfov(...))`。x 直接对原始 ERP boundary 使用 `minimalCircularInterval()`，y 直接取 min/max，从而消除 BFoV 包络后的第二次外接损失。旧间接 bbox 仍作为诊断保存，并计算 `envelopeInflation=indirectArea/directArea`。

## ERP bbox 到 BFoV

初始化框中心先转成球面点，水平/垂直像素跨度按 ERP 角分辨率转成角尺寸。跨缝框的中心使用循环区间，不使用普通 `(x1+x2)/2`。

## 数值边界

yaw 必须 wrap，pitch 必须 clamp，向量必须重新归一化，FOV 必须位于 (0,π)。极点附近优先使用向量计算，避免 yaw 在一个像素移动下大幅变化。

