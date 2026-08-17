# 深度处理算法

实现位于 `tracker/depth_preprocessor.py` 和 `tracker/depth_encoder.py`。

## 有效性与归一化

输入可以是 `DepthPlane` 或 float32 二维数组。算法先组合有限值、非负值和原始 validMask，得到有效像素。有效比例低于 `depth.minValidRatio` 时，摘要返回空并触发 RGB 回退。

归一化不会直接使用全局固定最大深度，而是依据有效值范围和背景估计构造相对深度表示，降低不同场景量程差异。无效像素保持掩码，不参与中位数和置信度。

## Relief 伪彩色

`mode=relief` 将相对深度映射为亮度/色相，同时计算平滑后的局部梯度作为边缘强度。`reliefGain` 控制深度起伏，`edgeGain` 强调物体边界，`nearBrightness/farBrightness` 限制远近亮度范围，最后转换为三通道 uint8 RGB。

`mode=grayscale` 只把归一化深度复制到三通道，适合验证颜色设计是否真正带来收益。`smoothingKernel` 必须是正奇数，过大会抹掉小目标边界，过小会放大传感噪声。

## 深度摘要

对目标 bbox 内的有效深度计算 medianDepth、有效比例和置信度。中位数对局部离群值比均值稳定。`maxDepthJumpRatio` 用于判断跨帧深度突变是否仍可信。

## 深度编码器

DepthEncoder 把 depthRgb 交给独立会话，并把返回值适配成统一 DepthFeatures/DepthPrediction。生产路径通过 `inferBatch()` 将本轮所有带深度的 depthRgb 按原 view 顺序组成一个 batch；底层 session 不支持 batch 时才逐图回退。模板与搜索必须使用相同的伪彩色配置，否则特征分布不一致。

## 优化观测

建议保存 normalized depth、depthRgb、有效掩码、目标摘要和最终 depthScore。只观察彩色图容易把“看起来清晰”误认为对 HiT 有帮助。

