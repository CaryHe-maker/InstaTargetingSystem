# AirSim360 数据接口方案

本文说明新的数据接口层。它把磁盘布局转换为项目统一的 `FramePacket`，因此跟踪器、几何模块、可视化和训练数据集都不需要知道输入文件的具体名字。

## 推荐的数据组织

以后增加测试用例时，建议把每个序列放在一个独立目录中：

```text
data/
  airsim360/
    nyc_0001/
      raw/                 # panorama_000000.png ...（RGB/RGBA 均可）
      depth/               # Depth_0.h5 ... 或同帧号 .npy/.png
      semantic/            # 同帧号语义 mask
      instance/            # 同帧号实例 mask
      semantic_lists.txt
```

文件只需共享末尾数字即可对齐，例如 `panorama_10.png`、`Depth_10.h5`、`10.png` 会组成 `frameIndex=10`。帧按数字而不是字符串排序，所以不会出现 `10` 排在 `2` 前面的错误。

规范目录使用 `raw/depth/semantic/instance`；为兼容旧项目，`rgb/depth/semantic/instance` 仍然有效。目录名可通过 `AirSim360SequenceSource` 的 alias 字段扩展，不需要修改跟踪代码。

## 解码规则

- RGB 读取为 `uint8[H,W,3]`；RGBA 会丢弃 alpha。
- `.h5/.hdf5` 深度使用 `readAirSim360DepthH5`，输出米制 `float32` 和显式 `validMask`。
- RGBA 语义图使用 alpha 通道作为类别 ID。
- RGBA 实例图把 RGB 以 `R | G<<8 | B<<16` 解码为 `int32` 实例 ID；单通道 PNG/NPY 保持原值，便于兼容已有标注。
- `semantic_lists.txt` 支持 `name id` 和 `id name` 两种行格式，结果放入 `SegmentationPlane.classNames`。
- 缺失模态保持 `None`，不会伪造零深度图；RGB、深度和 mask 若尺寸不一致会立即报告 `DecodeError/ProtocolError`。

## 程序接口

```python
from instatarget.data import openDataset

source = openDataset("data/airsim360/nyc_0001", format="auto")
try:
    while (frame := source.read()) is not None:
        # frame.rgb, frame.depth, frame.segmentation
        pass
finally:
    source.close()
```

等价的底层入口是 `AirSim360DataSource.open(root, sequenceId=None)`。传入包含多个序列的父目录时可以指定 `sequenceId`；传入单序列目录时省略它即可。`registerDatasetFormat("my_format", factory)` 为以后不同比赛输入注册新的适配器，适配器只需要实现 `open/read/close` 并产生 `FramePacket`。

## 目标选择

目标实例必须显式指定，不再自动选择。RGBA instance mask 的 RGB 三通道会解码为 packed ID，例如 `--target-instance 14198374`。该 ID 用于首帧生成初始框，后续帧由 tracker 持续跟踪同一个目标。语义类别 ID 不是 instance ID；要测试另一个物体，需要先获得它的 instance ID。

## 跟踪、测试输出和中间可视化

一条命令会生成最终结果、IoU 评估、最终绿色框图片和中间阶段可视化：

```bash
python tools/run_airsim360_dataset.py \
  --dataset-root data/airsim360/nyc_sample \
  --config configs/RGBonly.yaml \
  --output-dir artifacts/airsim360_smoke \
  --target-instance 14198374 \
  --max-frames 1
```

输出包括：

```text
artifacts/airsim360_smoke/
  result/
    manifest.json
    tracking.txt
    iou.json
    visualResult/frame_000000.png
  midVisual/<sequence>/frame_000000/{local_rgb,depth_rgb,backend_box,geometry_box}/
```

`result/visualResult/` 是最终提交框的绿色框可视化；`midVisual/` 保存局部裁剪、深度、backend 候选框和 geometry 候选框。`result/iou.json` 输出逐帧 IoU 以及 `meanIoU`、`successRate@0.5`、`auc`，并按 ERP seam 拆分后计算。

不传 `--output-dir` 时，工具会根据数据集相对 `data/` 的路径自动分配输出目录，并选择下一个编号：

```text
data/airsim360/nyc_sample/
  -> artifacts/airsim360/nyc_sample/output_1/
  -> artifacts/airsim360/nyc_sample/output_2/
```

如果显式传入 `--output-dir`，则使用用户指定的位置，不参与自动编号。

也可以直接调用：

```bash
python -m instatarget.track_airsim360 \
  --dataset-root data/airsim360 \
  --sequence nyc_0001 \
  --target-instance 14198374 \
  --output artifacts/nyc_0001/result.txt \
  --config configs/RGBD.yaml \
  --mid-visual-root artifacts/nyc_0001/midVisual \
  --result-visual-root artifacts/nyc_0001/result/visualResult
```

`--max-frames` 只用于快速回归；不传时会自动跑完整序列。当前后端没有加载第三方权重时使用确定性的 fallback HiT，因此仍可验证 I/O、几何、结果写出和可视化链路。

## 训练接口

`AirSim360TrainingDataset` 是不绑定 PyTorch 的懒加载数据集：

```python
from instatarget.training import AirSim360TrainingDataset

dataset = AirSim360TrainingDataset(
    "data/airsim360/nyc_0001", targetInstanceId=14198374
)
for sample in dataset:
    rgb = sample.rgb
    depth = sample.frame.depth
    box, visible = sample.targetBox, sample.visible
```

每个 `TrainingSample` 同时保留原始 `FramePacket`、伪标注框和可见性，后续可在训练脚本中转换为 Torch/ONNX tensor；数据层不把 I/O 绑定到某个框架。首帧实例框由跨 ERP seam 的最小圆周区间生成，适合 360° 图像。

## 扩展和数据质量检查

新增格式建议实现独立的 `DatasetSource`，完成以下检查后再注册：帧号唯一、RGB 与所有保留模态尺寸一致、深度单位明确、缺失文件显式为空。这样比赛最终输入格式变化时只需增加一个 adapter，不会修改 `runTracking`、控制器或模型训练代码。
