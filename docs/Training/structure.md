# Training 模块结构

Training 当前完整实现了样本读取契约，但训练循环、损失和配置仍是占位。文档明确区分“已实现”和“建议实现”。

| 文件 | 当前状态 |
|---|---|
| `training/dataset.py` | 已实现 AirSim360TrainingDataset |
| `training/losses.py` | TODO 占位 |
| `training/train_backend.py` | TODO 占位 |
| `configs/train_backend.yaml` | TODO 占位 |

深入阅读：[trainingDataset.md](trainingDataset.md)、[backendTraining.md](backendTraining.md)、[calibrationDataset.md](calibrationDataset.md)。

