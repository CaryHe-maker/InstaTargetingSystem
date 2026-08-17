# Tracker 参数索引

## 模型

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `model.backend` | `pytorch` | 会话后端；ONNX/TensorRT 尚不是完整生产路径 |
| `model.variant` | `hit_small` | HiT 结构变体 |
| `model.weights` | `../models/hit_small.pth` | checkpoint 路径 |
| `model.precision` | RGB `fp32` / RGB-D `fp16` | PyTorch HiT 推理精度；fp16 非有限时整批以 fp32 重算 |

## 深度

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `depth.enabled` | RGB false / RGB-D true | 是否创建深度分支 |
| `depth.minValidRatio` | 0.35 | 目标/视图深度最低有效比例 |
| `depth.maxDepthJumpRatio` | 0.60 | 跨帧深度突变容忍度 |
| `depth.colorization.mode` | `relief` | relief 或 grayscale |
| `nearBrightness` | 0.95 | 近处伪彩色亮度 |
| `farBrightness` | 0.20 | 远处伪彩色亮度 |
| `reliefGain` | 1.00 | 深度起伏强度 |
| `edgeGain` | 0.35 | 深度边缘增强 |
| `smoothingKernel` | 7 | 边缘/背景平滑核，必须为正奇数 |

## RGB-D 融合

| 参数 | RGB / RGB-D | 作用 |
|---|---:|---|
| `backendFusion.depthScoreWeight` | 0.0 / 0.35 | 最终局部分数中的深度质量 |
| `fusionHead.rgbInitWeight` | 0.70 | 剩余质量中的 RGB 比例基础 |
| `fusionHead.depthInitWeight` | 0.20 | 构造 FusionHead 的默认深度权重 |
| `fusionHead.contextInitWeight` | 0.10 | 剩余质量中的上下文比例基础 |
| `decisionGate.motionScoreWeight` | 0.25 | 旧聚合兼容权重 |
| `decisionGate.scaleScoreWeight` | 0.15 | 旧聚合兼容权重 |
| `decisionGate.depthConsistencyWeight` | 0.10 | 旧聚合兼容权重 |

深度缺失时无论权重如何都应回退 RGB 分数。调整融合权重后必须重新检查外观 Beta Calibration，因为 backend 原始 fusedScore 分布已经变化；生产候选最终再与有效运动概率按 70/30 合成 SingleScore。

