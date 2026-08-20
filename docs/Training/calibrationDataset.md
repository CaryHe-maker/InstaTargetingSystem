# 外观与运动校准数据流程

外观 Beta Calibration 与 SingleScore 权重由 Controller 模块拥有，但参数质量取决于离线数据，因此在 Training 模块记录统一的数据构建流程。唯一真实数据根是 `E:\NewDownload\train`；仓库 `data/` 不得用于校准、IoU 或发布指标。

## 样本收集

在冻结模型权重后运行 `tools/collect_score_calibration.py`。工具用生产 Geometry 与 Stage 3 backend 收集稳定 oracle views，保存 presence、quality、二者乘积、运动分、局部/球面真值误差及投影诊断。标签依据局部预测回投后的真实命中生成，而不是依据最终 Controller 是否输出，避免 Controller 漂移污染校准标签。

## 数据划分

校准集必须与模型训练集和最终测试集分离，并按序列划分。大量连续相邻帧会高度相关，可按序列/场景加权，避免长序列支配拟合。

## 拟合与选择

`tools/fit_score_calibration.py` 对 `presence*quality` 拟合 `sigmoid(c + alpha log(p) - beta log(1-p))` 三参数并约束单调。A1 同时比较 presence、quality 与乘积；当前乘积在 Brier、ECE 与 ROC-AUC 上最优。运动分当前保持已有同帧视图中心先验，不做逐帧 min-max。

## 与阈值联调

当前 E02 在 6 个 calibration 序列、4792 个候选上联合选择 50/50 SingleScore 与两个工作点：`candidateMinScore=0.597262`、`fusionSourceMinConfidence=0.740642`。`evaluator.successRate` 当前只写入诊断字段，不是生产门限。最终 holdout 尚未读取，也不能用于继续拟合或调阈值。

## 发布

发布参数存于版本化 JSON，并记录 checkpoint SHA-256、manifest SHA-256、输入字段、拟合参数、组合权重、阈值和指标。Runtime 严格验证产物结构、checkpoint 哈希和 YAML 阈值；任何模型替换都必须生成新产物。

