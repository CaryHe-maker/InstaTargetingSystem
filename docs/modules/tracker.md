# Tracker Backend

当前实现只有一个 HiT。`RGBonly.yaml` 直接把几何裁剪出的 RGB 局部图送入 HiT；`RGBD.yaml` 先让 backend 从对齐深度图预测边缘，再只修改这些边缘像素的 RGB 颜色，随后仍把这一张增强 RGB 图送入同一个 HiT。RGBD 不再有第二个 HiT、深度编码器或融合头。

## 输入输出

`TrackerBackend.initialize(template, templateBox)` 编码一份增强后的模板。`infer(views, command)` 按输入顺序返回 `LocalObservation`，并原子执行 `KEEP`、`UPDATE_RECENT`、`UPDATE_STABLE` 和 `RESET_TO_ANCHOR`。

两种模式都输出同一字段：`bbox`、`modelScore`、`appearanceScore`、`depthScore`、`fusedScore`、`depthSummary` 和 `latencyNs`。当前分数契约是：

- `fusedScore` 是单 HiT 的置信度，RGB-only 与 RGBD 都满足 `fusedScore == appearanceScore`。
- `depthScore` 不再代表第二模型分数，固定为 `0.0`。
- 深度有效时仍生成 `DepthSummary`，供控制层做几何/一致性门控；深度无效时摘要为 `None`。

## RGBD 边缘增强

`DepthPreprocessor.preprocess()` 归一化深度并计算梯度边缘。`depth.edge.threshold` 选择边缘，`depth.edge.widthPx` 膨胀边缘宽度，`depth.edge.minContrast` 控制边缘改色的最小 RGB 欧氏距离。

`enhanceRgb(rgb, depth)` 只写入预测边缘位置。非边缘像素逐字节保持原图；边缘像素选择原色的互补色或黑/白高反差色。增强图同时用于模板编码、HiT 搜索、在线模板更新和可视化。

## 真实 HiT 会话

生产 driver 通过 `model.backend=pytorch`、`model.source`、`model.variant`、`model.weights`、`model.precision` 和 `model.device` 创建官方 `kangben258/HiT` 会话。源码目录必须包含官方 `lib/` 与 `experiments/HiT/`；权重缺失、依赖缺失或 CUDA 不可用都会抛出 `ModelError`，不会静默退回伪模型。测试可以显式注入 `FallbackHiTSession`。

官方 HiT 原始输出没有分类概率，因此 adapter 使用 query 一致性、模板间框一致性和状态转移稳定度计算 `[0,1]` 的 `appearanceScore`，并将其作为 `fuseScore`。

## 边界

backend 不生成 BFoV、不规划搜索、不修改控制状态。geometry 负责 RGB/Depth 同步裁剪，DTC 负责多视图规划、候选聚合和模板命令决策。
