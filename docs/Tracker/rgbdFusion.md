# RGB-D 分数融合

## 两个独立会话

RGB-D 模式不是把深度直接作为第四通道送给 RGB HiT。Runtime 建立两个独立 HiT 会话：RGB 会话处理原局部图，Depth 会话处理深度伪彩色图。这样能复用现有三通道模型结构，并让深度缺失时自然退化到 RGB-only。

## 单视图处理

对每个 LocalView：

1. RGB 会话输出 bbox、modelScore、appearanceScore。
2. 若深度存在且有效，DepthPreprocessor 生成 depthRgb，深度会话用对应模板特征输出 depthScore。
3. Controller 预测中心与局部候选中心形成 context/motion 证据。
4. FusionHead 合并 RGB、深度和上下文分数，得到 LocalObservation.fusedScore。

框几何仍由 RGB HiT 输出；深度当前主要修正置信度，不单独提出一个不同 bbox。

## 融合公式

当深度可用时，`backendFusion.depthScoreWeight=d` 直接占据总质量的一部分；剩余质量按 `rgbInitWeight` 和 `contextInitWeight` 比例分配。最后再除以实际权重和，输出裁剪到 [0,1]。

当深度不可用或 d=0 时，FusionHead 直接返回 RGB 分数，而不是把缺失深度当作 0 分参与平均。这一点防止 RGB-only 被无意义压低。

## 参数作用

- `backendFusion.depthScoreWeight`：深度分支对最终分数的显式强度，RGB-only 为 0。
- `fusionHead.rgbInitWeight/depthInitWeight/contextInitWeight`：初始化与兼容权重。
- `decisionGate.*Weight`：旧聚合兼容路径的运动、尺度、深度权重，不等同于 FusionHead 主公式。

## 优化方法

应分别统计 RGB 正确/深度错误、RGB 错误/深度正确、两者一致和深度缺失四类样本。只在整体平均 IoU 上调 d，可能掩盖远距离稀疏深度对高分目标的伤害。

