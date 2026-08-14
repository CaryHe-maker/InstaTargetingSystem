# AirSim360 数据处理方案

本文说明 AirSim360 数据在 InstaTargetingSystem 中的读取、目标选择、跟踪与评估流程。数据接口与比赛视频接口共用 `FramePacket`、球面几何、控制器和 HiT 后端，输出格式根据运行入口分别适配。

## 数据读取

`AirSim360SequenceSource` 按帧匹配 RGB、深度、语义分割和实例分割文件。RGB 是必需输入；其余模态按文件是否存在写入 `FramePacket`。RGB-only 配置允许帧中携带深度数据，但后端不会建立深度处理器，也不会使用深度参与推理或决策。

支持的目录别名如下：

| 模态 | 目录名 |
|---|---|
| RGB | `rgb`、`raw` |
| 深度 | `depth`、`Depth` |
| 语义分割 | `semantic`、`segmentation` |
| 实例分割 | `instance`、`instances` |

读取器也支持 `meta.json` 中的显式 `records` 列表。深度文件支持 HDF5、NumPy 数组和图像；RGBA 分割图的 Alpha 通道用于语义 ID，RGB 三通道按 24 位整数组合为实例 ID。

## 目标初始化

目标由整数 `instance ID` 指定。`PseudoTrackBuilder` 在第 0 帧实例掩码中计算目标框，并使用水平循环区间处理跨越 ERP 左右边界的目标。第 0 帧框作为初始化输入，不计入 IoU 汇总中的模型预测帧。

实例清单可通过一条命令生成：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" --dataset-root "data\airsim360\nyc_sample" --output "artifacts\easy_user\nyc_sample\InstanceID.txt"
```

## 跟踪与评估

`tools/run_airsim360_dataset.py` 完成以下操作：

1. 读取第 0 帧并生成初始化框。
2. 按配置创建真实 HiT-Small 运行时。
3. 顺序处理所有帧并写入 `tracking.txt`。
4. 由实例掩码生成逐帧伪真值并计算循环边界框 IoU。
5. 写入清单、指标和可视化产物。

RGB-only 与 RGB-D 的完整一行命令见 [User/EasyUser.md](User/EasyUser.md)。输出目录结构和指标定义见 [Verification.md](Verification.md)。

## 模态行为

RGB-only 使用一个 HiT-Small 会话处理局部 RGB 视图。RGB-D 使用两个相互独立的 HiT-Small 会话：RGB 会话处理彩色图像，深度会话处理深度预处理器生成的伪彩色三通道图像；融合模块将模型、外观、深度和控制器证据组合为候选分数。

深度图缺失或有效比例不足时，深度摘要为空，控制器仍可依据 RGB、运动、尺度和多视角一致性提交结果。
