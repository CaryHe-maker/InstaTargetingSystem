# 分数校准

## 位置与目的

`controller/fused_score.py` 统一拥有外观校准、运动残差概率和 SingleScore 合成。调用分两步：Tracker 后、Geometry 前生成 `appearanceProbability`；局部框回投后生成运动概率和 `singleScore`。StateEvaluator 的单框排序、来源门限和双框融合只使用 `singleScore`。

HiT 原始高分过于集中时，直接使用会让 0.80 与 0.95 看起来都接近成功。Beta Calibration 将高分整体压低，并拉开该区间的差距。

## 映射

```text
calibrated = sigmoid(c + alpha*log(p) - beta*log(1-p))
```

当前参数 `FUSED_SCORE_BETA_PARAMETERS=(14.30532301, 1.52758886, -2.21085783)`，对应锚点 0.80→0.05、0.90→0.45、0.98→0.97。新映射把 0.80–0.98 拉伸到 0.05–0.97，解决 backend 高分集中且候选间差距过小的问题。p=0 和 p=1 保持端点，非法或非有限分数直接报协议错误。校准结果写入 `appearanceProbability`，不会覆盖 backend 原始 `fusedScore`。

## MotionScore 与 SingleScore

Runtime 使用 `scoreViewCenterMotion()` 计算同帧局部视图运动先验：预测中心为 1.0，每增加 30 度大圆夹角连续下降 0.1。它以局部图中心而不是检测框中心计量，不经过 reliability 向 0.5 混合，也不做当前帧 min-max。`scoreMotionConsistency()` 的协方差归一化中心/尺度/深度残差保留用于离线诊断；当前 motion calibration 仍为 identity。

```text
singleScore = clip(0.70*appearanceProbability + 0.30*effectiveMotion, 0, 1)
```

`ProjectedObservation.backendFusedScore` 保存 backend 原值，`appearanceProbability` 保存 Beta 结果；`rawMotionScore/motionProbability/motionScore` 在 Runtime 路径中保存同一个局部视图中心角度分，`motionReliability` 只保留预测器成熟度诊断，不缩放视图间分数。`singleScore` 是控制器使用的最终单框分数。兼容字段 `ProjectedObservation.fusedScore` 当前同步为 `singleScore`，新代码不应依赖该别名。

## 轮次评分

所有轮次都使用固定的 70/30 `SingleScore`。`SearchPlan.appearanceOnlyScoring` 仍作为协议兼容字段保留，但当前固定为 `false`；正常线程的 TRACKING/UNCERTAIN 在第二轮提交时把两轮观测合并后统一按 `singleScore` 排序和融合。保留的显式 LOST 组件仍把 6 张 cubemap 和 4 张 Type1 在同一轮统一交给 Fusor。

## 为什么不用分段映射

Beta 形式连续、可微、单调，且能独立调节 p 和 1-p 两端形状。分段线性映射在断点处斜率突变，容易让很小的模型分数抖动跨过 StateEvaluator 阈值。

## 重新拟合

当前外观、运动和 70/30 合成参数都是代码常量，不是 YAML 超参数。正式重新标定应在独立验证集同时收集 backend 原分、`d2/rawMotionScore`、命中标签和投影质量；外观比较 Beta Calibration，运动比较 Beta/isotonic/固定 logit-temperature，再冻结参数。不能用测试集 IoU 手调单个参数，否则会发生数据泄漏。

StateEvaluator 的提交与融合工作点不是新的分数映射：外观 Beta 映射及其锚点保持不变。状态机动态门限只读取已提交的 `StateScore` 历史，因此不会改变 `fusedScore -> appearanceProbability` 的跨帧语义。

评估时至少检查 reliability diagram、Brier score、负对数似然、候选排序 AUC，以及 0.8–0.95 区间排序是否保持。还要联合扫描 80/20、70/30、60/40 与 `tracking.candidateMinScore/fusionSourceMinConfidence`，因为校准改变了有效门限的实际含义。`evaluator.successRate` 当前只用于诊断记录，不参与生产决策。

