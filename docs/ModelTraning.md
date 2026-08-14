# 训练数据接口

仓库提供与运行时数据契约一致的训练数据读取层，便于复用 RGB、深度和实例标注。该文件描述数据接口和仓库边界，不定义一个独立的训练执行命令。

## `AirSim360TrainingDataset`

构造参数包括数据根目录、可选序列、目标实例 ID 和格式选择。目标实例 ID 必须显式提供，以保证样本生成具有确定性。迭代返回 `TrainingSample`：

| 字段 | 内容 |
|---|---|
| `frame` | 对齐的 `FramePacket` |
| `targetInstanceId` | 目标实例整数 ID |
| `targetBox` | 由实例掩码计算的 ERP 框 |
| `visible` | 当前帧是否包含目标 |

样本层使用 NumPy 数组，不绑定深度学习框架。伪真值由 `PseudoTrackBuilder` 计算，目标不可见时保留框对象并将 `visible` 设为 `False`。

## 数据要求

- RGB 文件必须可读取且尺寸一致。
- 实例掩码必须与 RGB 对齐。
- 深度是可选模态；存在时必须满足 `DepthPlane` 的类型、单位和有效掩码约束。
- 目标实例 ID 必须在第 0 帧中存在。

模型训练循环、优化器和权重发布不属于本仓库的运行时提交路径。比赛与评估使用现有 `models/hit_small.pth`，由 [HiTRuntime.md](HiTRuntime.md) 中的适配器加载。
