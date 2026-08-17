# 运动评分与边界回投执行方案

本文把运动预测评分和 ViewSpec 回投精度两项改进合并为同一条生产链，描述当前已经落地的行为、验证标准和后续边界。

## 已执行方案

1. Geometry 对局部框四边各采样 `boundarySamplesPerEdge` 个点，只投影一次。
2. 同一组球面边界通过区间中点迭代修正中心，再用最终水平/垂直角区间宽度拟合无旋转 BFoV。
3. ERP bbox 直接从原始边界点拟合：x 使用最小循环区间，y 使用 min/max；不再经过 BFoV 的第二次边界包络。
4. MotionPredictor 保留球面切平面 Huber 常速度估计，同时输出中心、log 尺度、可选深度的协方差和可靠性。
5. Runtime 按局部图中心与当前帧预测中心的大圆夹角计算视图运动先验：0 度为 1.0，每增加 30 度连续下降 0.1；协方差归一化候选残差保留为离线诊断。
6. 运动历史只有一个样本时使用按历史成熟度降权的位置/尺度锚点；首个非空弱候选最多提供一次启动样本。正常序列从第三个输入帧起具备速度样本，不再因硬开关长期回退到中性 0.5。
7. backend 原始融合分只做外观 Beta Calibration，不被覆盖。最终单框分数固定为 `0.70*appearanceProbability + 0.30*effectiveMotionProbability`。
8. StateEvaluator 只用 `singleScore` 做单框排序、来源门限和双框融合；旧 `fusedScore` 仅保留兼容回退。

## 字段流

```text
LocalObservation.fusedScore
  -> backendFusedScore
  -> appearanceProbability

MotionPrediction center + local ViewSpec center
  -> great-circle angle
  -> same-frame effectiveMotion

appearanceProbability + effective motion
  -> singleScore
  -> StateEvaluator
```

`ProjectedObservation` 同时保存原始 ERP boundary、`envelopeInflation`、`normalizedRadius` 和 `edgeMargin`，用于区分模型边界误差、投影边缘畸变和二次包络损失。

## 当前冻结参数

- 外观 Beta 参数：`(14.30532301, 1.52758886, -2.21085783)`，锚点为 0.80→0.05、0.90→0.45、0.98→0.97。
- SingleScore 权重：appearance 0.70，motion 0.30。
- 运动组合权重：尺度残差 0.35，深度残差 0.15，`d2` 上限 25。
- motion calibration 当前为 identity；在独立校准集完成前，不使用逐帧 min-max 或伪造经验参数。

所有轮次都使用 70/30 SingleScore。LOST 第一轮的两个旋转 cubemap 在同一批次内评估；TRACKING/UNCERTAIN 第一轮先由 Fusor 选择中心，第二轮围绕该中心推理 VStype1 四角 4 张视图，提交时将两轮投影观测统一交给 Fusor。`appearanceOnlyScoring` 作为兼容字段固定为 false。

## 验证门槛

单元测试必须覆盖视图中心、边缘、高纬度、ERP 经线、运动历史不足、可靠预测、尺度偏差和可选深度。端到端实验至少报告 BFoV spherical IoU、ERP circular IoU、中心角误差、宽高相对误差、候选排序 AUC、Brier score、`envelopeInflation`、每状态帧数和 P95 延迟。

## 尚未实施

窄 FOV 二次 refinement、384/512 模型输入、mask refinement、旋转框和重新训练均需要额外模型推理或训练资产，当前生产链未启用。下一阶段应先用现有诊断字段证明误差来源，再独立评估这些方案，不能把它们写成当前能力。
