# 分数校准

## Stage 3 生产合同

生产模型为 `models/hit_small_stage3.pth`。Tracker 保留 Stage 3 的 `presence` 与 `quality`（预测 IoU），Runtime 以二者乘积作为外观原始概率：

```text
rawAppearance = clip(presence * predictedIoU, 0, 1)
appearanceProbability = sigmoid(c + alpha*log(p) - beta*log(1-p))
singleScore = clip(0.50*appearanceProbability + 0.50*effectiveMotion, 0, 1)
```

presence、quality 与二者乘积已在同一 calibration 数据上做 A1 比较。乘积的 Brier、ECE 与 ROC-AUC 最好，因此生产路径选择乘积，不是为兼容旧模型而保留的规则。

当前校准来自 `models/hit_small_stage3.calibration.json`：`alpha=0.9934308915`、`beta=1.8582728356`、`intercept=0.6623364310`。它使用 `E:\NewDownload\train\manifest.jsonl` 的 6 个 calibration 序列、4792 个候选拟合，Brier 从 `0.17895` 降至 `0.13769`，ECE 从 `0.18271` 降至 `0.01685`，PR-AUC 为 `0.89583`，ROC-AUC 为 `0.88040`。

## 产物绑定

`buildRuntime()` 在生产启动时必须加载校准 JSON。严格加载器拒绝未知/缺失字段、非有限参数、非单调 Beta 参数、非 calibration split、错误输入类型以及非法 SHA-256。默认还会核对 checkpoint 内容哈希，并要求 JSON 中的两个工作点与 YAML 完全一致：

- `tracking.candidateMinScore = 0.597262`
- `evaluator.fusionSourceMinConfidence = 0.740642`

缺少校准产物时生产运行直接失败。identity 校准只允许评估工具通过显式 `--uncalibrated-stage3` 执行预校准 E01，不能用于正常入口。旧手工 Beta、旧 checkpoint 别名和旧分数 remap 已删除。

## 字段语义

`backendFusedScore` 保存 Stage 3 的 `presence * predictedIoU` 原值，`appearanceProbability` 保存校准结果。`rawMotionScore/motionProbability/motionScore` 在当前 Runtime 路径保存同一局部视图中心角度分；`motionReliability` 只作诊断。`singleScore` 是 StateEvaluator 排序、来源门限和融合使用的唯一单框分数。

所有 TRACKING/UNCERTAIN 轮次使用相同校准。第二轮提交时，两轮观测放入同一个候选池；保留但正常线程不可达的显式 LOST 组件也使用同一评分合同。校准或权重变化时必须生成新版本产物并重新选择两个工作点，禁止复用当前 JSON。
