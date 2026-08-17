# ViewSpec 与回投框边缘精度建议

> 本文仅提供分析和实施方案，不修改 Geometry、ViewSpec、Tracker 或结果格式代码。

## 结论

当前边缘膨胀不一定主要来自“局部图分辨率不够”。从现有链路看，更应先检查两次几何包络：

```text
local bbox boundary
  -> 拟合一个对称、无旋转的 BFoV 包络
  -> 再投影 BFoV 边界
  -> 取 ERP axis-aligned 外接 bbox
```

第一次拟合可能扩大 BFoV，第二次外接矩形又可能在曲边四角引入空白。建议先修正表示和回投路径，再考虑 512 分辨率、锐化或超分辨率。

## 当前链路中的具体放大来源

Runtime 当前执行：

```text
candidateBfov = geometry.localBoxToBfov(localBox, viewSpec)
erpBbox = geometry.bfovToBbox(candidateBfov, frameWidth, frameHeight)
```

`localBoxToBfov()` 先采样局部框四条边，然后 `_fitBfovFromVectors()`：

1. 用边界向量均值作为中心；
2. 在该中心切平面计算所有点的水平/垂直角；
3. 使用 `2 * max(abs(angle))` 作为 FOV。

如果边界相对均值中心不完全对称，`2*max(abs)` 会把较短一侧扩展到与较长一侧相同。随后 `bfovToBbox()` 又对这个已经扩展的 BFoV 边界取 ERP 最小循环外接矩形，因此可能二次膨胀。

此外，透视局部矩形映到 ERP 后通常是曲边四边形。任何 axis-aligned ERP 外接矩形都必然包含一些不属于目标投影的角落；纬度越高、局部 FOV 越大、目标越靠局部视图边缘，这个问题越明显。

## 第一优先级：拆分 BFoV 与 ERP bbox 的拟合路径

建议新增“一次直接回投”的几何结果：

```text
projectLocalBoxBoundary(localBox, viewSpec)
    -> sphericalBoundary / erpBoundary

fitBfovFromBoundary(sphericalBoundary)
fitErpBboxDirect(erpBoundary)
```

ERP bbox 不应再通过 `bfovToBbox(localBoxToBfov(...))` 间接产生，而应直接对原始局部框边界映射点计算：

- x 使用跨经线安全的 `minimalCircularInterval`；
- y 使用原始边界投影点的 min/max；
- 保留投影 polygon，供诊断、可视化和更精确 IoU 使用。

这不会消除 axis-aligned bbox 的固有空角，但能消除“BFoV 包络后再包络”的额外损失。

需要同时记录：

```text
envelopeInflation = indirectBboxArea / directBoundaryBboxArea
```

并按纬度、ViewSpec FOV、局部框中心到视图中心的距离分组。

## 第二优先级：改进 BFoV 拟合

比赛最终提交的是四字段无旋转 BFoV，因此仅缩小内部 ERP bbox 不会自动改善比赛 BFoV IoU。BFoV 本身需要更紧。

建议不要固定“均值中心 + 对称最大绝对角”。可参考 360VOT 的 `localBbox2Bfov`：先把局部框边界映到球面，在合适的局部坐标中计算 min/max 或最小面积旋转矩形，再由区间中点反算中心。

无旋转 BFoV 的暂定迭代算法：

1. 以边界向量均值建立初始切平面。
2. 计算水平角 `[minH,maxH]` 和垂直角 `[minV,maxV]`。
3. 用 `(minH+maxH)/2`、`(minV+maxV)/2` 修正球面中心，而不是强迫围绕旧中心对称。
4. 在新中心重复 2–3 次，直到中心变化小于阈值。
5. 最终 FOV 使用 `maxH-minH` 和 `maxV-minV`。

如果评测格式未来允许 rotation，则 rBFoV/最小面积旋转框通常能进一步减少空白；当前四字段提交不应擅自加入 rotation。

## 第三优先级：120° 搜索与窄 FOV 精修分离

120° FOV 有利于召回，但 gnomonic perspective 在边缘拉伸明显，不适合做最后边缘定位。建议保留现有 120° 视图作为 coarse search，在候选出现后增加一个 refinement ViewSpec：

```text
refineCenter = candidate BFoV center
refineFovH = clamp(contextFactor * candidateFovH, 30°, 75°)
refineFovV = clamp(contextFactor * candidateFovV, 30°, 75°)
contextFactor 初始测试 2.0–3.0
```

候选重新置于局部图中央后，用同一个 HiT 或专门 bbox refinement head 再预测一次。相比单纯把局部图从 256 放大到 512，缩小 FOV 会直接增加“每个目标角度对应的模型输入像素数”，在固定 `_SEARCH_SIZE=256` 下更有效。

该方案增加一次 backend 推理，适合只对以下候选触发：

- 将要作为最终结果的最高分候选；
- 局部框中心位于视图外侧 30% 区域；
- 预测的几何 inflation ratio 超阈值；
- 两个视图中心一致但尺寸差异明显。

## 分辨率与长宽比例

### 是否提高到 384/512

可以实验，但当前 HiT 会把任意搜索图 resize 回 256×256。因此只改 `ViewSpec.outputWidthPx/outputHeightPx` 会增加 Geometry 采样成本，却不会让网络看到更多像素。真正使用 384/512 需要同步修改 HiT 输入、位置编码/模型配置，并最好在相同输入尺寸上微调。

建议顺序：

1. 256 + 120° baseline；
2. 256 + 窄 FOV refinement；
3. 384/512 模型输入 + 微调；
4. 再判断超分辨率是否仍有必要。

### 是否拉宽或拉长局部图

当前后端把搜索图强制 resize 为正方形。非正方形 ViewSpec 会被压缩/拉伸到 256×256，使目标比例发生变化，因此不建议只修改局部图长宽比。若确需矩形输入，模型、训练增强、归一化框转换和模板路径都要一起支持 preserve-aspect letterbox。

## 视图边缘质量参与选择

同一目标常出现在多个重叠局部视图中。建议为每个局部框记录投影质量：

```text
normalizedRadius = distance(localBoxCenter, viewCenter) / viewHalfDiagonal
edgeMargin = min(distance to four local image edges) / imageSize
jacobianDistortion = local projection area scale at box center
```

这些量不应直接替代外观分数，但可以：

- 优先选择更靠近局部视图中心的框作为 representative；
- 对严重边缘框请求 refinement；
- 在两个分数接近的候选中作为 tie-breaker；
- 分析框膨胀到底来自模型还是投影。

## 是否使用最大内接矩形

不建议把“最大被投影区域包含的 axis-aligned 矩形”作为默认最终框。它会主动舍弃目标真实边缘：precision 可能提高，但 recall 必然下降；当真值是完整目标外接框时，IoU 不一定改善。

它可以作为诊断上界或短期对照实验：

- 外接矩形：保证覆盖，偏大；
- 最大内接矩形：保证不超出，偏小；
- validation-tuned quantile box：在两者之间。

若必须做临时 shrink，建议按 `latitude × localNormalizedRadius × FOV` 在验证集拟合四边独立收缩量，而不是全局固定乘 0.9。正式方案仍应优先修复 BFoV 拟合和增加窄 FOV refinement。

## 更高精度的长期方案

1. **Mask refinement**：对最高候选运行轻量分割头，将局部 mask 逐像素回投球面，再用 360VOT 类似的 `mask2Bfov/mask2Bbox` 拟合最终框。这最接近真实物体边缘，但训练和时延成本最高。
2. **Rotated representation**：内部保留 rBBox/rBFoV，最后只在输出边界降级成比赛允许的无旋转 BFoV。
3. **IoU-aware bbox head**：在本项目局部透视裁剪分布上微调 bbox head，并加入 IoU/边界质量预测，避免只依赖 heatmap certainty。
4. **多视图边界融合**：不要只交叉两个 ERP axis-aligned bbox；可先将两个局部框回投为球面 polygon，再求 polygon 交集/稳健中位边界，最后拟合一次 BFoV。

## 验证顺序

1. 实现离线诊断：原始边界 polygon、当前 BFoV、间接 ERP bbox、直接 ERP bbox同时绘制。
2. 测量两次包络的 inflation ratio，确认主要误差来源。
3. 单独测试“非对称 BFoV 拟合”，不改变 Tracker。
4. 单独测试“直接 ERP bbox”，区分内部 bbox 指标和比赛 BFoV 指标。
5. 测试窄 FOV refinement，并统计额外延迟。
6. 最后比较 384/512、mask refinement 和临时 shrink。

测试集必须覆盖：视图中心、视图边缘、ERP 经线、赤道、高纬度/极点、大目标、小目标和部分出视野目标。指标至少包括 BFoV spherical IoU、ERP circular bbox IoU、中心角误差、宽高相对误差、目标像素数、inflation ratio 和 P95 延迟。

## 参考资料

- [360VOT toolkit](https://github.com/HuajianUP/360VOT)：提供 BBox、rBBox、BFoV、rBFoV 四种表示，以及 `localBbox2Bfov`、`localBbox2Bbox`、`mask2Bfov`、`mask2Bbox` 的球面转换参考。
- [360VOT omni.py](https://github.com/HuajianUP/360VOT/blob/main/lib/omni.py)：可直接对照局部框边界到全景 bbox/BFoV 的实现方式。
- [HiT official repository](https://github.com/kangben258/HiT)：核对模型输入尺寸、训练配置和 bbox head 微调边界。

