# 分数校准

## 位置与目的

`controller/fused_score.py` 在 Tracker 推理后、Geometry 回投前处理每个 `LocalObservation.fusedScore`。校准发生在双框融合之前，因此 StateEvaluator 的单框和融合公式都使用校准后的分数。

HiT 原始高分过于集中时，直接使用会让 0.80 与 0.95 看起来都接近成功。Beta Calibration 将高分整体压低，并拉开该区间的差距。

## 映射

```text
calibrated = sigmoid(c + alpha*log(p) - beta*log(1-p))
```

当前参数 `FUSED_SCORE_BETA_PARAMETERS=(7.62702021, 0.91697230, -1.50849067)`，对应锚点 0.80→0.15、0.90→0.45、0.95→0.70。p=0 和 p=1 保持端点，非法或非有限分数直接报协议错误。

## 为什么不用分段映射

Beta 形式连续、可微、单调，且能独立调节 p 和 1-p 两端形状。分段线性映射在断点处斜率突变，容易让很小的模型分数抖动跨过 StateEvaluator 阈值。

## 重新拟合

当前参数是代码常量，不是 YAML 超参数。正式重新标定应收集独立验证集上的原始分数与命中标签，用 Beta Calibration 最大似然拟合三参数，再冻结整组三元组。不能用测试集 IoU 手调单个参数，否则会发生数据泄漏。

评估时至少检查 reliability diagram、Brier score、负对数似然，以及 0.8–0.95 区间排序是否保持。还要联动评估 `successRate`，因为校准改变了阈值的实际含义。

