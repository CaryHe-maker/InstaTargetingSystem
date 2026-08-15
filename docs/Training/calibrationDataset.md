# Beta Calibration 数据流程

Beta Calibration 本身运行在 Controller，但参数质量取决于离线数据，因此在 Training 模块记录数据构建流程。

## 样本收集

在冻结模型权重和 Tracker 融合设置后运行独立校准序列，保存每个局部视图的原始 fusedScore、投影框、真值框、状态、round 和 view role。标签应依据局部预测投影后的真实命中标准生成，而不是依据最终 Controller 是否输出。

## 数据划分

校准集必须与模型训练集和最终测试集分离，并按序列划分。大量连续相邻帧会高度相关，可按序列/场景加权，避免长序列支配拟合。

## 拟合与选择

拟合 `sigmoid(c + alpha log(p) - beta log(1-p))` 三参数，约束映射单调。用负对数似然或 Brier score 选择参数，同时查看 reliability diagram 和高分段样本数。若 0.8–0.95 区间样本不足，不能仅靠三个手选锚点宣称泛化良好。

## 与阈值联调

先冻结校准映射，再在另一验证子集选择 `successRate` 和来源最低置信度。把同一数据同时用于拟合映射和调阈值会产生过拟合。

## 发布

当前参数存于 `controller/fused_score.py::FUSED_SCORE_BETA_PARAMETERS`。替换时应保存数据版本、模型权重哈希、拟合脚本、三参数、校准指标和端到端回归结果。

