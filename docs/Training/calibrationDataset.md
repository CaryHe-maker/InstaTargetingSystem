# 外观与运动校准数据流程

外观 Beta Calibration、运动校准和 SingleScore 权重都由 Controller 模块拥有，但参数质量取决于离线数据，因此在 Training 模块记录统一的数据构建流程。

## 样本收集

在冻结模型权重和 Tracker 融合设置后运行独立校准序列，保存每个候选的 `backendFusedScore`、`appearanceProbability`、中心/尺度/深度残差、预测协方差、`rawMotionScore`、reliability、SingleScore、球面/ERP 真值误差、状态、round、目标纬度、view role、`normalizedRadius`、`edgeMargin` 和 `envelopeInflation`。标签应依据局部预测回投后的真实命中标准生成，而不是依据最终 Controller 是否输出。

## 数据划分

校准集必须与模型训练集和最终测试集分离，并按序列划分。大量连续相邻帧会高度相关，可按序列/场景加权，避免长序列支配拟合。

## 拟合与选择

外观拟合 `sigmoid(c + alpha log(p) - beta log(1-p))` 三参数并约束单调。运动分数比较 Beta Calibration、isotonic regression 和固定 logit-temperature；样本不足时继续使用 identity，不做逐帧 min-max。用负对数似然、Brier score、候选排序 AUC 和 reliability diagram 选择参数。若高分段或遮挡/边缘样本不足，不能仅靠手选锚点宣称泛化良好。

## 与阈值联调

先冻结两个校准映射，再在另一验证子集联合比较 80/20、70/30、60/40，并选择 `tracking.candidateMinScore` 和 `evaluator.fusionSourceMinConfidence`。`evaluator.successRate` 当前只写入诊断字段，不是可调生产门限。把同一数据同时用于拟合映射和调阈值会产生过拟合。

## 发布

当前参数和 SingleScore 权重存于 `controller/fused_score.py`。替换时应保存数据版本、模型权重哈希、拟合脚本、外观/运动参数、组合权重、校准指标和端到端回归结果。

