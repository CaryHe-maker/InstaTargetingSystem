# Tracker 参数索引

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `model.backend` | `pytorch` | 会话后端；ONNX/TensorRT 尚不是完整生产路径 |
| `model.variant` | `hit_small` | HiT 结构变体 |
| `model.weights` | `../models/hit_small_stage3.pth` | Stage 3 checkpoint 路径 |
| `model.precision` | `fp32` | PyTorch HiT 推理精度；fp16 非有限时整批以 fp32 重算 |

TrackerBackend 从 Stage 3 输出 `presence`、`quality/predictedIoU` 和 bbox，并以 `presence*quality` 作为 backend `fusedScore`。Runtime 使用 checkpoint 绑定校准产物映射外观概率，再与有效运动概率按产物冻结的 50/50 权重合成 SingleScore。修改模型权重后必须重新拟合校准并重选候选门限，不能复用旧 checkpoint 的参数。
