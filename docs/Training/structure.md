# Training 模块结构

Training 现在包含 manifest 驱动的数据对、生产 Geometry crop、训练损失、HiT 显式置信度头、分阶段冻结策略和可恢复训练循环。

| 文件 | 当前状态 |
|---|---|
| `training/dataset.py` | AirSim360TrainingDataset + ManifestPairDataset/索引解码 |
| `training/manifest_builder.py` | BFoV groundtruth 转 manifest、sequence split |
| `training/augment.py` | RGB 域增强，保持局部框语义 |
| `training/model.py` | HiT wrapper、presence/quality heads、Stage 冻结 |
| `training/losses.py` | mask-aware presence/L1/GIoU/quality loss |
| `training/train_backend.py` | AMP、累积梯度、分层 optimizer、验证、checkpoint/resume |
| `configs/train_backend.yaml` | 以冻结 Stage 3 权重为起点的严格后续微调配置 |
| `tools/collect_score_calibration.py` | Stage 3 oracle-view calibration 候选收集 |
| `tools/fit_score_calibration.py` | 单调 Beta、SingleScore 权重和工作点拟合 |

深入阅读：[trainingDataset.md](trainingDataset.md)、[backendTraining.md](backendTraining.md)、[calibrationDataset.md](calibrationDataset.md)。

