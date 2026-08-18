# Tracker 参数索引

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `model.backend` | `pytorch` | 会话后端；ONNX/TensorRT 尚不是完整生产路径 |
| `model.variant` | `hit_small` | HiT 结构变体 |
| `model.weights` | `../models/hit_small.pth` | checkpoint 路径 |
| `model.precision` | `fp32` | PyTorch HiT 推理精度；fp16 非有限时整批以 fp32 重算 |

TrackerBackend 直接使用 HiT 的 `appearanceScore` 作为 backend `fusedScore`。Runtime 随后执行外观 Beta Calibration，并与有效运动概率按 70/30 合成 SingleScore。修改模型权重或精度后必须重新检查原始分数分布、外观校准和候选门限。
