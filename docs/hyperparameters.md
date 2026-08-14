# 配置与超参数

运行配置采用 `schemaVersion: 1` 的严格 YAML 模式。缺失字段、未知字段、类型错误或越界值都会触发 `ConfigError`。相对路径以配置文件所在目录为基准解析。

## 标准配置

| 文件 | 模态 | 精度 | HiT 会话数 |
|---|---|---|---:|
| `configs/RGBonly.yaml` | RGB-only | FP32 | 1 |
| `configs/RGBD.yaml` | RGB-D | FP16，异常时同模型 FP32 重算 | 2 |

生产配置固定使用 `model.backend: pytorch`、`model.variant: hit_small` 和 `models/hit_small.pth`。

## 字段分组

| 分组 | 作用 |
|---|---|
| `model` | 后端、模型变体、权重路径和数值精度 |
| `geometry` | 局部视图尺寸、边界采样数和视场角范围 |
| `depth` | 深度开关、有效率阈值、跳变阈值和伪彩色参数 |
| `backendFusion` | 深度分数在后端融合中的权重 |
| `fusionHead` | RGB、深度和上下文的初始化权重 |
| `decisionGate` | 运动、尺度和深度一致性权重 |
| `evaluator` | 多视角支持度、一致性和重捕获最小视图数 |
| `motion` | 球面运动估计窗口、噪声和速度限制 |
| `tracking` | 接纳阈值、状态耐心、同帧升级和视图预算 |
| `recovery` | 环形搜索、立方体扫描和覆盖记录参数 |
| `runtime` | 受校验的容量元数据 |
| `visualization` | 诊断输出开关、根目录和阶段集合 |

## 关键约束

- `tracking.uncertainThreshold < tracking.acceptThreshold`。
- `tracking.recoverAcceptThreshold >= tracking.acceptThreshold`。
- `tracking.maxAttemptsPerFrame` 只能为 1 或 2。
- `tracking.maxViewsPerFrameTotal` 至少覆盖六个立方体面及 `minViewsForCommit`。
- `geometry.minFovDeg` 与 `geometry.maxFovDeg` 满足 `0 < min < max < 180`。
- `depth.colorization.smoothingKernel` 为正奇数。
- `visualization.stages` 只能包含 `local_rgb`、`depth_rgb`、`backend_box`、`geometry_box`。
- 比赛配置要求 `depth.enabled: false` 且 `backendFusion.depthScoreWeight: 0.0`。

## 容量字段说明

`decodeQueueCapacity`、`inferRequestQueueCapacity`、`inferResponseQueueCapacity` 和 `resultQueueCapacity` 必须为正整数。它们属于配置模式的一部分；`runTracking()` 采用同步逐帧调度，不据此创建队列。
