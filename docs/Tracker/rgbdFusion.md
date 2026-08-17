# RGB-D 分数融合

## 两个独立会话

RGB-D 模式不是把深度直接作为第四通道送给 RGB HiT。Runtime 建立两个独立 HiT 会话：RGB 会话处理原局部图，Depth 会话处理深度伪彩色图。这样能复用现有三通道模型结构，并让深度缺失时自然退化到 RGB-only。

## 本轮批量处理

Tracker 先对本轮全部 LocalView 执行一个 RGB batch，再把其中带深度的视图组成一个 depth batch。两个会话按此顺序调用，不是同时运行；各自结果按原始 view 顺序重新对齐。随后对每个 LocalView：

1. 从 RGB batch 取得 bbox、modelScore、appearanceScore。
2. 若深度存在，DepthPreprocessor 生成 depthRgb，并从 depth batch 取得 depthScore；目标区域摘要无效时该分数按 0 处理并回退 RGB。
3. FusionHead 合并 RGB、深度和上下文分数，得到 backend 原始 `LocalObservation.fusedScore`。
4. Runtime 在回投前生成 `appearanceProbability`，回投后再由 Controller 的校准模块结合运动残差生成 `singleScore`。

框几何仍由 RGB HiT 输出；即使 depth session 返回 bbox，当前也只使用其分数，不让深度分支提出不同几何。

## 融合公式

当深度可用时，`backendFusion.depthScoreWeight=d` 直接占据 backend 总质量的一部分；剩余质量按 `rgbInitWeight` 和 `contextInitWeight` 比例分配。最后再除以实际权重和，输出裁剪到 [0,1]。这个结果是外观/模态证据，不是 StateEvaluator 的最终单框分数。

当深度不可用或 d=0 时，FusionHead 直接返回 RGB 分数，而不是把缺失深度当作 0 分参与平均。这一点防止 RGB-only 被无意义压低。

## 参数作用

- `backendFusion.depthScoreWeight`：深度分支对最终分数的显式强度，RGB-only 为 0。
- `fusionHead.rgbInitWeight/depthInitWeight/contextInitWeight`：初始化与兼容权重。
- `decisionGate.*Weight`：旧聚合兼容路径的运动、尺度、深度权重，不等同于 FusionHead 主公式。
- SingleScore 的 0.70 外观/0.30 运动权重属于 Controller 校准模块，不属于 Tracker FusionHead。

## 优化方法

应分别统计 RGB 正确/深度错误、RGB 错误/深度正确、两者一致和深度缺失四类样本。只在整体平均 IoU 上调 d，可能掩盖远距离稀疏深度对高分目标的伤害。

